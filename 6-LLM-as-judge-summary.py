import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import statistics

# ===========================
# Summary Generation Script
# ===========================
EVALUATED_FOLDER = "datasets"  # Base folder containing dataset/model evaluations
OUTPUT_DIR = "plots"            # Directory to store summary
MODELS_TO_SUMMARIZE = ["gemma3_12b", "llama3.1_8b"]  # Corresponding folder names after ':' replaced with '_'

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def summarize_evaluations():
    summary_data = defaultdict(lambda: defaultdict(list))

    # Traverse datasets and models
    for dataset in os.listdir(EVALUATED_FOLDER):
        dataset_path = Path(EVALUATED_FOLDER) / dataset / "qrels"
        if not dataset_path.exists():
            continue

        for model_name in MODELS_TO_SUMMARIZE:
            model_path = dataset_path / model_name
            if not model_path.exists():
                continue

            # Find all evaluated JSON files
            for file in model_path.glob("*_evaluated.json"):
                print(f"📄 Processing file: {file.name}")
                with open(file, "r", encoding="utf-8") as f:
                    records = json.load(f)

                # Collect metrics
                for record in records:
                    eval_json = record.get("evaluation_json", "{}")
                    if not eval_json:
                        continue
                    try:
                        eval_dict = json.loads(eval_json)
                    except json.JSONDecodeError:
                        continue

                    for q_col, metrics in eval_dict.items():
                        if metrics:
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    summary_data[(dataset, model_name, q_col, file.name)][key].append(value)

    # Aggregate results (mean + variance)
    aggregated_summary = []
    for key, metrics_dict in summary_data.items():
        dataset, model_name, question_type, filename = key
        aggregated_metrics = {}
        for metric, vals in metrics_dict.items():
            if vals:
                mean_val = sum(vals) / len(vals)
                var_val = statistics.variance(vals) if len(vals) > 1 else 0.0
                aggregated_metrics[f"{metric}_mean"] = mean_val
                aggregated_metrics[f"{metric}_var"] = var_val
            else:
                aggregated_metrics[f"{metric}_mean"] = None
                aggregated_metrics[f"{metric}_var"] = None

        aggregated_summary.append({
            "dataset": dataset,
            "model": model_name,
            "question_type": question_type,
            "source_file": filename,
            **aggregated_metrics
        })

    # Convert to DataFrame
    summary_df = pd.DataFrame(aggregated_summary)
    summary_csv_path = Path(OUTPUT_DIR) / "evaluation_summary.csv"
    summary_json_path = Path(OUTPUT_DIR) / "evaluation_summary.json"

    summary_df.to_csv(summary_csv_path, index=False)
    summary_df.to_json(summary_json_path, orient="records", indent=2)

    print(f"✅ Summary saved to {summary_csv_path} and {summary_json_path}")

if __name__ == "__main__":
    summarize_evaluations()
