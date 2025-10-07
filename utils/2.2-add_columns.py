import pandas as pd
from pathlib import Path

# Base directory
base_dir = Path("/Users/marco/Documents/GitHub/dataset_distribution_evaluation/datasets/hotpotqa/qrels")

# Leggi i file originali (test e dev)
test_original = pd.read_csv(base_dir / "test_text.tsv", sep=",")
dev_original = pd.read_csv(base_dir / "dev_text.tsv", sep=",")

# Crea dizionari query per lookup veloce
test_queries = dict(zip(test_original["query-id"], test_original["query"]))
dev_queries = dict(zip(dev_original["query-id"], dev_original["query"]))

# Itera sulle cartelle dei modelli
for model_dir in base_dir.iterdir():
    if not model_dir.is_dir():
        continue
    print(f"\n📂 Model folder: {model_dir.name}")

    # --- test_text.tsv ---
    test_target = model_dir / "test_text.tsv"
    if test_target.exists():
        df_test = pd.read_csv(test_target, sep="\t")

        if "query-id" in df_test.columns:
            df_test["query"] = df_test["query-id"].map(test_queries)
            missing = df_test["query"].isna().sum()
            if missing > 0:
                print(f"⚠️ {missing} query non trovate nel test originale.")
        else:
            print(f"❌ 'query-id' non trovato in {test_target.name}")

        df_test.to_csv(test_target, sep="\t", index=False)
        print(f"💾 Updated test file: {test_target}")
    else:
        print(f"⚠️ Missing: {test_target}")

    # --- dev_text.tsv ---
    dev_target = model_dir / "dev_text.tsv"
    if dev_target.exists():
        df_dev = pd.read_csv(dev_target, sep="\t")

        if "query-id" in df_dev.columns:
            df_dev["query"] = df_dev["query-id"].map(dev_queries)
            missing = df_dev["query"].isna().sum()
            if missing > 0:
                print(f"⚠️ {missing} query non trovate nel dev originale.")
        else:
            print(f"❌ 'query-id' non trovato in {dev_target.name}")

        df_dev.to_csv(dev_target, sep="\t", index=False)
        print(f"💾 Updated dev file: {dev_target}")
    else:
        print(f"⚠️ Missing: {dev_target}")
