import os
import uuid
import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import ollama

from classes.classes_prompt import *
from classes.classes_server_ollama import *

# ===========================
# Dataset & Loop Config
# ===========================
DATASETS_FOLDER = "datasets"
QRELS_SUBFOLDER = "qrels"
MODELS_LLM = ["phi3:14b"]
BATCH_SIZE = 200

# Only these datasets, in strict order
DATASETS_TO_USE = ["fiqa"]

# Single prompt file near code
PROMPT_FILE_PATH = Path("generated_prompt.txt")

# Initialize server
ollama_server = OllamaServer()
print("Available Models:", ollama_server.get_models_list())

# ===========================
# Main Loop (Refactored)
# ===========================
for model_name in MODELS_LLM:
    print(f"\n🚀 Running model: {model_name}")
    chat = OllamaChat(server=ollama_server, model=model_name)

    for dataset in DATASETS_TO_USE:
        dataset_path = os.path.join(DATASETS_FOLDER, dataset)
        dataset_qrels_path = os.path.join(dataset_path, QRELS_SUBFOLDER)

        if not os.path.isdir(dataset_qrels_path):
            print(f"⚠️ Skipping {dataset} (no qrels folder found).")
            continue

        # Load few-shot examples
        fewshot_sample_path = os.path.join(dataset_path, "samples/sample_1.tsv")
        if not os.path.exists(fewshot_sample_path):
            print(f"⚠️ No sample_1.tsv found for {dataset}, skipping.")
            continue

        df_fewshot = pd.read_csv(fewshot_sample_path, sep="\t")
        df_fewshot.columns = df_fewshot.columns.str.strip()

        fewshot_examples = []
        for _, row in df_fewshot.iterrows():
            fewshot_examples.append({
                'good_question': row.get('query', ''),
                'bad_question': row.get('bad_query', ''),
                'motivation': row.get('motivation', '')
            })

        passage_col_fewshot = next(
            (c for c in df_fewshot.columns if c.lower() in ["passage", "text", "document", "content"]),
            None
        )
        if passage_col_fewshot is None:
            print(f"⚠️ No passage column in {dataset} few-shot, skipping.")
            continue

        # Build prompts
        prompt_obj = Prompt_Creation(df_fewshot[passage_col_fewshot], fewshot_examples)
        inpair_prompt = prompt_obj.create_prompt_inpair()
        base_prompt = prompt_obj.create_prompt_base()
        motivation_prompt = prompt_obj.create_prompt_inpair_motivation()

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

            passage_col = next((c for c in df.columns if c.lower() in ["passage", "text", "document", "content"]), None)
            if passage_col is None:
                print(f"No passage column in {filename}, skipping.")
                continue

            if output_file_path.exists():
                df_saved = pd.read_csv(output_file_path, sep="\t")
                start_idx = len(df_saved)
                predicted_inpair_questions = df_saved.get('predicted_inpair_query', []).tolist()
                predicted_base_questions = df_saved.get('predicted_base_query', []).tolist()
                predicted_motivation_questions = df_saved.get('predicted_inpair_motivation_query', []).tolist()
            else:
                start_idx = 0
                predicted_inpair_questions = []
                predicted_base_questions = []
                predicted_motivation_questions = []

            # Prepare question generators
            question_gen_inpair = Question_Creation(chat, inpair_prompt)
            question_gen_base = Question_Creation(chat, base_prompt)
            question_gen_motivation = Question_Creation(chat, motivation_prompt, return_motivation=True)

            # Process passages
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                               desc=f"{dataset} → {filename} ({model_name})"), start=1):
                if idx <= start_idx:
                    continue

                passage = row.get(passage_col, "")
                if not isinstance(passage, str) or not passage.strip():
                    predicted_inpair_questions.append("No question available")
                    predicted_base_questions.append("No question available")
                    predicted_motivation_questions.append("No question available")
                    continue

                # Use named placeholder to match new prompt format
                q_inpair = question_gen_inpair.extract_question(passage=passage)
                q_base = question_gen_base.extract_question(passage=passage)
                q_motivation, _ = question_gen_motivation.extract_question(passage=passage)

                predicted_inpair_questions.append(q_inpair)
                predicted_base_questions.append(q_base)
                predicted_motivation_questions.append(q_motivation)

                # Save periodically
                if (idx - start_idx) % BATCH_SIZE == 0:
                    df_temp = df.iloc[:idx].copy()
                    df_temp['predicted_inpair_query'] = predicted_inpair_questions
                    df_temp['predicted_base_query'] = predicted_base_questions
                    df_temp['predicted_inpair_motivation_query'] = predicted_motivation_questions
                    df_temp.to_csv(output_file_path, sep="\t", index=False)

            # Save final results
            df['predicted_inpair_query'] = predicted_inpair_questions
            df['predicted_base_query'] = predicted_base_questions
            df['predicted_inpair_motivation_query'] = predicted_motivation_questions
            df.to_csv(output_file_path, sep="\t", index=False)
