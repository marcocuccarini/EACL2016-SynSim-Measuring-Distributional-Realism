import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import ollama
import uuid

from classes.classes_server_ollama import *

# ===========================
# Question Generation Class
# ===========================
class Question_Creation:
    def __init__(self, chat, prompt_template, return_motivation=False):
        self.chat = chat
        self.prompt_template = prompt_template
        self.return_motivation = return_motivation

    def extract_question(self, passage1, passage2=None, print_prompt=False):
        """
        Generate a question using two passages and the prompt template.
        If print_prompt=True, prints the final prompt before sending to the model.
        """
        prompt = self.prompt_template.replace("{passage1}", passage1)
        if passage2 is not None:
            prompt = prompt.replace("{passage2}", passage2)
        
        if print_prompt:
            print("\n--- Prompt Sent to Model ---")
            print(prompt)
            print("--- End of Prompt ---\n")
        
        response = self.chat.send_prompt(prompt)
        return response.raw_text

# ===========================
# Dual-Passage Prompt Class
# ===========================
class Prompt_Creation_Dual:
    def __init__(self, fewshot_examples):
        self.fewshot_examples = fewshot_examples

    def create_prompt_base(self):
        prompt = (
            "You are a precise question-generation assistant.\n"
            "Given **two passages**, generate **ONE clear, concise question** that meaningfully connects both passages.\n"
            "**Only output the question—do not add explanations or extra text.**\n\n"
        )
        for ex in self.fewshot_examples:
            prompt += (
                f"Passage 1: {ex['passage1']}\n"
                f"Passage 2: {ex['passage2']}\n"
                f"Good question: {ex['query']}\n\n"
            )
        prompt += (
            "Now generate a question for the following passages:\n"
            "Passage 1: {passage1}\n"
            "Passage 2: {passage2}\n"
            "Question:"
        )
        return prompt

    def create_prompt_inpair(self):
        prompt = (
            "You are a precise question-generation assistant.\n"
            "Given **two passages**, write **ONE high-quality question** that connects both passages.\n"
            "Use the examples below to see the difference between good and bad questions.\n"
            "**Only output the question—do not add explanations or extra text.**\n\n"
        )
        for ex in self.fewshot_examples:
            prompt += (
                f"Passage 1: {ex['passage1']}\n"
                f"Passage 2: {ex['passage2']}\n"
                f"Good question: {ex['query']}\n"
                f"Bad question: {ex['bad_query']}\n\n"
            )
        prompt += (
            "Now generate a question for the following passages:\n"
            "Passage 1: {passage1}\n"
            "Passage 2: {passage2}\n"
            "Question:"
        )
        return prompt

    def create_prompt_inpair_motivation(self):
        prompt = (
            "You are a precise question-generation assistant.\n"
            "Given **two passages**, write **ONE high-quality question** that connects both passages.\n"
            "Use the examples below, which include bad questions and explanations describing why they are inadequate.\n"
            "**For new passages, only output the good question—do not write explanations.**\n\n"
        )
        for ex in self.fewshot_examples:
            prompt += (
                f"Passage 1: {ex['passage1']}\n"
                f"Passage 2: {ex['passage2']}\n"
                f"Good question: {ex['query']}\n"
                f"Bad question: {ex['bad_query']}\n"
                f"Why the bad question is inadequate: {ex['motivation']}\n\n"
            )
        prompt += (
            "Now generate a question for the following passages:\n"
            "Passage 1: {passage1}\n"
            "Passage 2: {passage2}\n"
            "Question:"
        )
        return prompt

# ===========================
# Dataset & Loop Config
# ===========================
DATASETS_FOLDER = "datasets"
QRELS_SUBFOLDER = "qrels"
MODELS_LLM = ["llama3.1:8b"]
BATCH_SIZE = 500
DATASETS_TO_USE = ["hotpotqa"]

# Initialize server
ollama_server = OllamaServer()
print("Available Models:", ollama_server.get_models_list())

# ===========================
# Main Loop
# ===========================
prompt_dumped = False  # Only dump prompts once

for model_name in MODELS_LLM:
    print(f"\n🚀 Running model: {model_name}")
    chat = OllamaChat(server=ollama_server, model=model_name)

    for dataset in DATASETS_TO_USE:
        dataset_path = os.path.join(DATASETS_FOLDER, dataset)
        dataset_qrels_path = os.path.join(dataset_path, QRELS_SUBFOLDER)

        if not os.path.isdir(dataset_qrels_path):
            print(f"⚠️ Skipping {dataset} (no qrels folder found).")
            continue

        fewshot_sample_path = os.path.join(dataset_path, "samples/sample_1.tsv")
        if not os.path.exists(fewshot_sample_path):
            print(f"⚠️ No sample_1.tsv found for {dataset}, skipping.")
            continue

        # Load few-shot examples
        df_fewshot = pd.read_csv(fewshot_sample_path, sep="\t")
        df_fewshot.columns = df_fewshot.columns.str.strip()
        fewshot_examples = []
        for _, row in df_fewshot.iterrows():
            fewshot_examples.append({
                'passage1': row.get('passage1', ''),
                'passage2': row.get('passage2', ''),
                'query': row.get('query', ''),
                'bad_query': row.get('bad_query', ''),
                'motivation': row.get('motivation', '')
            })

        # Initialize prompts
        prompt_obj = Prompt_Creation_Dual(fewshot_examples)
        inpair_prompt = prompt_obj.create_prompt_inpair()
        base_prompt = prompt_obj.create_prompt_base()
        motivation_prompt = prompt_obj.create_prompt_inpair_motivation()

        # Dump prompts to a file the first time
        if not prompt_dumped:
            prompt_file = Path(dataset_path) / f"{model_name.replace(':','_')}_sample_prompt.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write("=== BASE PROMPT ===\n")
                f.write(base_prompt.replace("{passage1}", "DUMMY_PASSAGE1").replace("{passage2}", "DUMMY_PASSAGE2") + "\n\n")
                f.write("=== INPAIR PROMPT ===\n")
                f.write(inpair_prompt.replace("{passage1}", "DUMMY_PASSAGE1").replace("{passage2}", "DUMMY_PASSAGE2") + "\n\n")
                f.write("=== MOTIVATION PROMPT ===\n")
                f.write(motivation_prompt.replace("{passage1}", "DUMMY_PASSAGE1").replace("{passage2}", "DUMMY_PASSAGE2") + "\n")
            print(f"💾 Sample prompts dumped to {prompt_file}")
            prompt_dumped = True

        # Clear chat history
        chat.clear_history()

        similarity_files = [f for f in os.listdir(dataset_qrels_path) if f.endswith("_text.tsv")]
        if not similarity_files:
            print(f"⚠️ Skipping {dataset} (no *_text.tsv files found).")
            continue

        for filename in similarity_files:
            file_path = os.path.join(dataset_qrels_path, filename)
            model_output_dir = Path(dataset_qrels_path) / model_name.replace(":", "_")
            model_output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = model_output_dir / filename

            df = pd.read_csv(file_path, sep=",")
            df.columns = df.columns.str.strip()

            # Prepare question generators
            question_gen_inpair = Question_Creation(chat, inpair_prompt)
            question_gen_base = Question_Creation(chat, base_prompt)
            question_gen_motivation = Question_Creation(chat, motivation_prompt)

            # Group passages by query-id
            grouped = df.groupby('query-id')
            output_rows = []
            processed_pairs = 0

            # ✅ Resume logic: check if output file already exists
            completed_query_ids = set()
            if os.path.exists(output_file_path):
                try:
                    existing_df = pd.read_csv(output_file_path, sep="\t")
                    completed_query_ids = set(existing_df['query-id'].unique())
                    output_rows = existing_df.to_dict(orient='records')
                    print(f"🔄 Resuming {filename}: found {len(completed_query_ids)} completed query-ids.")
                except Exception as e:
                    print(f"⚠️ Could not read existing file ({e}), starting fresh.")

            # Process each query group
            for query_id, group in tqdm(grouped, desc=f"{dataset} → {filename} ({model_name})"):
                if len(group) < 2:
                    continue

                # ⏩ Skip if this query-id was already processed
                if query_id in completed_query_ids:
                    continue

                passage1 = group.iloc[0]['passage']
                passage2 = group.iloc[1]['passage']

                if not all(isinstance(p, str) and p.strip() for p in [passage1, passage2]):
                    continue

                # Original query (if available in CSV)
                original_query = group.iloc[0].get('query', '')

                # Generate questions
                q_inpair = question_gen_inpair.extract_question(passage1, passage2)
                q_base = question_gen_base.extract_question(passage1, passage2)
                q_motivation = question_gen_motivation.extract_question(passage1, passage2)

                # Add two rows (one per passage)
                output_rows.append({
                    'query-id': query_id,
                    'passage': passage1,
                    'original_query': original_query,
                    'predicted_inpair_query': q_inpair,
                    'predicted_base_query': q_base,
                    'predicted_inpair_motivation_query': q_motivation
                })
                output_rows.append({
                    'query-id': query_id,
                    'passage': passage2,
                    'original_query': original_query,
                    'predicted_inpair_query': q_inpair,
                    'predicted_base_query': q_base,
                    'predicted_inpair_motivation_query': q_motivation
                })

                processed_pairs += 1

                # Batch save every BATCH_SIZE pairs
                if processed_pairs % BATCH_SIZE == 0:
                    df_out = pd.DataFrame(output_rows)
                    tmp_path = output_file_path.with_suffix(".tmp")
                    df_out.to_csv(tmp_path, sep="\t", index=False)
                    os.replace(tmp_path, output_file_path)
                    print(f"💾 Batch saved after {processed_pairs} pairs for {filename}")

            # Final save
            df_out = pd.DataFrame(output_rows)
            tmp_path = output_file_path.with_suffix(".tmp")
            df_out.to_csv(tmp_path, sep="\t", index=False)
            os.replace(tmp_path, output_file_path)
            print(f"✅ Final save completed for {filename} ({len(df_out)} rows)")

print("🎉 All processing completed.")
