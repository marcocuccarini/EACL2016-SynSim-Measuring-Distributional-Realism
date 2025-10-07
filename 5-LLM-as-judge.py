import os
import uuid
import datetime
import re
import time
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import ollama

from classes.classes_server_ollama import *

# ===========================
# LLM Judge: Single JSON per question
# ===========================
class LLMJudgeSingleJSON:
    """
    Judge that evaluates each question on all five metrics at once
    and returns a single JSON object.
    """

    def __init__(self, chat: OllamaChat):
        self.chat = chat
        self.chat.clear_history()

    def _make_prompt(self, passage: str, question: str) -> str:
        few_shot_examples = """
Example 1:
Passage: "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France."
Question: "Where is the Eiffel Tower located?"
JSON Output:
{
  "relevance": 5,
  "clarity": 5,
  "answerability": 5,
  "difficulty": 2,
  "overall_score": 5
}

Example 2:
Passage: "Water boils at 100 degrees Celsius under standard pressure."
Question: "What is the capital of France?"
JSON Output:
{
  "relevance": 1,
  "clarity": 4,
  "answerability": 1,
  "difficulty": 1,
  "overall_score": 1
}

Example 3:
Passage: "Photosynthesis is how plants produce energy using sunlight."
Question: "Explain how plants make energy."
JSON Output:
{
  "relevance": 5,
  "clarity": 5,
  "answerability": 5,
  "difficulty": 3,
  "overall_score": 5
}

Example 4:
Passage: "Python is a popular programming language known for readability."
Question: "What are Python’s main design goals?"
JSON Output:
{
  "relevance": 4,
  "clarity": 5,
  "answerability": 4,
  "difficulty": 3,
  "overall_score": 4
}

Example 5:
Passage: "The sun rises in the east and sets in the west."
Question: "Describe the process of nuclear fusion in stars."
JSON Output:
{
  "relevance": 2,
  "clarity": 4,
  "answerability": 2,
  "difficulty": 4,
  "overall_score": 2
}
"""

        scale_description = """
Use the following 1–5 scale:
1 = Very poor / unrelated / unclear / not answerable
2 = Poor / weak / limited clarity / difficult
3 = Fair / somewhat relevant / moderate clarity
4 = Good / mostly relevant / clear / answerable
5 = Excellent / fully relevant / perfectly clear / fully answerable
"""

        prompt = f"""
You are an evaluation assistant.
Rate the following passage and question on these five metrics: relevance, clarity, answerability, difficulty, overall_score.
Return ONLY valid JSON following this schema:
{{"relevance": int, "clarity": int, "answerability": int, "difficulty": int, "overall_score": int}}

{scale_description}

Use the few-shot examples below as guidance:
{few_shot_examples}

Now evaluate:
Passage: "{passage}"
Question: "{question}"

Return only a single JSON object.
"""
        return prompt.strip()

    def evaluate(self, passage: str, question: str) -> dict:
        prompt = self._make_prompt(passage, question)
        response = self.chat.send_prompt(prompt, use_history=False)
        raw_text = response.raw_text

        print("\n--- Raw LLM output ---")
        print(raw_text)
        print("--- End of output ---\n")

        # Extract first JSON object
        match = re.search(r'\{\s*"relevance"[\s\S]*?\}', raw_text)
        if not match:
            print("⚠️ No valid JSON found.")
            return {}
        try:
            return json.loads(match.group())
        except:
            print("⚠️ Failed to parse JSON.")
            return {}

# ===========================
# Main Evaluation Script
# ===========================
DATASETS_FOLDER = "datasets"
QRELS_SUBFOLDER = "qrels"
DATASETS_TO_USE = ["fiqa", "nq"]
MODELS_TO_USE = ["llama3.1:8b"]
BATCH_SIZE = 200

def main():
    ollama_server = OllamaServer()
    available_models = ollama_server.get_models_list()
    print("Available Models:", available_models)

    models_to_run = MODELS_TO_USE if MODELS_TO_USE else available_models
    datasets_to_run = DATASETS_TO_USE if DATASETS_TO_USE else [
        d for d in os.listdir(DATASETS_FOLDER) if os.path.isdir(os.path.join(DATASETS_FOLDER, d))
    ]

    for dataset in datasets_to_run:
        dataset_path = os.path.join(DATASETS_FOLDER, dataset)
        dataset_qrels_path = os.path.join(dataset_path, QRELS_SUBFOLDER)

        if not os.path.exists(dataset_qrels_path):
            print(f"⚠️ No qrels folder found in {dataset_path}, skipping.")
            continue

        for model_name in models_to_run:
            model_dir_name = model_name.replace(":", "_")
            model_output_dir = Path(dataset_qrels_path) / model_dir_name

            if not model_output_dir.exists():
                print(f"⚠️ No model output folder for {dataset}/{model_name}, skipping.")
                continue

            print(f"\n🚀 Running single-question JSON judge for dataset: {dataset}, model: {model_name}")
            chat = OllamaChat(server=ollama_server, model=model_name)
            judge = LLMJudgeSingleJSON(chat)

            tsv_files = [f for f in os.listdir(model_output_dir) if f.endswith(".tsv")]
            if not tsv_files:
                print(f"⚠️ No TSV files found in {model_output_dir}, skipping.")
                continue

            for tsv_file in tsv_files:
                file_path = model_output_dir / tsv_file
                df = pd.read_csv(file_path, sep="\t")
                df.columns = df.columns.str.strip()

                passage_col = next((c for c in df.columns if c.lower() in ["passage", "text", "document", "content"]), None)
                if passage_col is None:
                    print(f"⚠️ No passage column in {tsv_file}, skipping.")
                    continue

                question_columns = [
                    "predicted_inpair_query",
                    "predicted_base_query",
                    "predicted_inpair_motivation_query"
                ]

                df["evaluation_json"] = None

                # Evaluate questions
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {tsv_file}"):
                    passage = row.get(passage_col, "")
                    if not isinstance(passage, str) or not passage.strip():
                        continue

                    all_results = {}
                    for q_col in question_columns:
                        question = row.get(q_col, "")
                        if not isinstance(question, str) or not question.strip():
                            continue

                        result = judge.evaluate(passage, question)
                        all_results[q_col] = result

                    df.at[idx, "evaluation_json"] = json.dumps(all_results, ensure_ascii=False)

                    if (idx + 1) % BATCH_SIZE == 0:
                        partial_file = model_output_dir / f"{tsv_file.replace('.tsv', '_partial.json')}"
                        df.to_json(partial_file, orient="records", indent=2, force_ascii=False)
                        print(f"💾 Saved partial progress after {idx + 1} rows → {partial_file}")

                # Save final JSON
                evaluated_json = model_output_dir / f"{tsv_file.replace('.tsv', '_evaluated.json')}"
                df.to_json(evaluated_json, orient="records", indent=2, force_ascii=False)
                print(f"✅ Final evaluations saved: {evaluated_json}")

    print("\n🎉 All datasets/models evaluated successfully.")

if __name__ == "__main__":
    main()
