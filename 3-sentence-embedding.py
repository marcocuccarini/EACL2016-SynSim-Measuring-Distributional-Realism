import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from pathlib import Path

# 🔹 Modelli di embedding locali
'''models = {
    "all-MiniLM-L6-v2": SentenceTransformer('all-MiniLM-L6-v2'),
    "paraphrase-MiniLM-L12-v2": SentenceTransformer('paraphrase-MiniLM-L12-v2'),
    "msmarco-distilbert-base-v4": SentenceTransformer('msmarco-distilbert-base-v4'),
    "e5-base-v2": SentenceTransformer("intfloat/e5-base-v2"),
    "mpnet-base": SentenceTransformer("microsoft/mpnet-base"),
}'''

models = {
    "all-MiniLM-L6-v2": SentenceTransformer('all-MiniLM-L6-v2')
}



def compute_similarities(file_path, model_name, model):
    """Calcola similarità query vs passage e aggiunge colonne con i punteggi"""
    print(f"Processing {file_path} with model {model_name}...")
    df = pd.read_csv(file_path, sep="\t")

    # Identifica colonne di query e passage
    query_cols = [col for col in df.columns if col.lower().endswith("query")]
    passage_cols = [col for col in df.columns if col.lower().endswith("passage")]

    for q_col in query_cols:
        for p_col in passage_cols:
            print(f"Comparing {q_col} vs {p_col}")

            queries = df[q_col].astype(str).tolist()
            passages = df[p_col].astype(str).tolist()

            # Calcolo embedding
            query_embeddings = model.encode(queries, convert_to_tensor=True)
            passage_embeddings = model.encode(passages, convert_to_tensor=True)

            # Similarità coseno
            similarities = util.cos_sim(query_embeddings, passage_embeddings)

            # Solo diagonale (query[i] vs passage[i])
            row_similarities = [similarities[i][i].item() for i in range(len(df))]

            # Aggiungi colonna
            col_name = f"sim_{q_col}_vs_{p_col}_{model_name}"
            df[col_name] = row_similarities

    return df


if __name__ == "__main__":
    datasets = ["fiqa", "hotpotqa", "nq"]

    for dataset in datasets:
        qrels_dir = Path("datasets") / dataset / "qrels"

        for llm_dir in qrels_dir.iterdir():
            if not llm_dir.is_dir():
                continue

            llm_name = llm_dir.name
            print(f"\n📂 Dataset: {dataset}, LLM: {llm_name}")

            for file_path in llm_dir.glob("*.tsv"):
                for model_name, model in models.items():
                    output_dir = llm_dir / f"sim_{model_name}"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    output_file = output_dir / file_path.name

                    # 🔒 Controllo robusto: esiste ed è non vuoto
                    if output_file.exists() and output_file.stat().st_size > 0:
                        print(f"⏭ Skipping {output_file}, già esiste ed è valido.")
                        continue

                    # Calcolo e salvataggio
                    df_sim = compute_similarities(file_path, model_name, model)
                    df_sim.to_csv(output_file, sep="\t", index=False)
                    print(f"💾 Saved to {output_file}")
