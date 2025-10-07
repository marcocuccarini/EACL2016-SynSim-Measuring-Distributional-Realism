import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde, entropy, wasserstein_distance, skew, kurtosis, jarque_bera
from pathlib import Path

# Setup
sns.set(style="whitegrid")

MODELS = [
    "sim_all-MiniLM-L6-v2",
    "sim_paraphrase-MiniLM-L12-v2",
    "sim_msmarco-distilbert-base-v4"
]

SIMILARITY_TYPES = {
    "Query vs Passage": "sim_query_vs_passage_{}",
    "Predicted Inpair Query vs Passage": "sim_predicted_inpair_query_vs_passage_{}",
    "Predicted Base Query vs Passage": "sim_predicted_base_query_vs_passage_{}",
    "Predicted Motivated Query vs Passage": "sim_predicted_inpair_motivation_query_vs_passage_{}"
}

# Desired order: Base → Inpair → Motivation → Query
SIMILARITY_ORDER = [
    "Predicted Base Query vs Passage",
    "Predicted Inpair Query vs Passage",
    "Predicted Motivated Query vs Passage",
    "Query vs Passage"
]

# --- Fixed color palette for consistency ---
QUERY_COLORS = {
    "Predicted Base Query vs Passage": "#1f77b4",      # blue
    "Predicted Inpair Query vs Passage": "#ff7f0e",    # orange
    "Predicted Motivated Query vs Passage": "#2ca02c", # green
    "Query vs Passage": "#d62728"                      # red
}

ROOT_DIR = Path("datasets")
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Functions ---

def compute_distribution_overlap(base_scores, compare_scores, method="ovl"):
    base_scores = np.array(base_scores)
    compare_scores = np.array(compare_scores)

    if method == "ovl":
        kde_base = gaussian_kde(base_scores)
        kde_compare = gaussian_kde(compare_scores)
        xs = np.linspace(0, 1, 1000)
        return np.trapz(np.minimum(kde_base(xs), kde_compare(xs)), xs)

    elif method == "jsd":
        hist_base, _ = np.histogram(base_scores, bins=100, range=(0, 1), density=True)
        hist_compare, _ = np.histogram(compare_scores, bins=100, range=(0, 1), density=True)
        hist_base += 1e-8
        hist_compare += 1e-8
        m = 0.5 * (hist_base + hist_compare)
        return 0.5 * entropy(hist_base, m) + 0.5 * entropy(hist_compare, m)

    elif method == "wasserstein":
        return wasserstein_distance(base_scores, compare_scores)

    else:
        raise ValueError(f"Unknown method: {method}")

def compute_normality_metrics(scores):
    scores = np.array(scores)
    if len(scores) < 3:
        return {"Skewness": np.nan, "Kurtosis": np.nan, "Jarque-Bera": np.nan}
    
    s = skew(scores)
    k = kurtosis(scores, fisher=False)
    jb_stat, jb_p = jarque_bera(scores)
    return {"Skewness": round(s, 4), "Kurtosis": round(k, 4), "Jarque-Bera": round(jb_stat, 4)}

# --- Main processing ---
overlap_results = []

for dataset_dir in ROOT_DIR.iterdir():
    if not dataset_dir.is_dir():
        continue
    dataset_name = dataset_dir.name
    qrels_dir = dataset_dir / "qrels"

    for llm_dir in qrels_dir.iterdir():
        if not llm_dir.is_dir():
            continue
        llm_name = llm_dir.name

        for tsv_file in llm_dir.glob("*.tsv"):
            file_base = tsv_file.stem

            for model in MODELS:
                model_file = llm_dir / model / tsv_file.name
                if not model_file.exists():
                    print(f"⚠️ File not found: {model_file}")
                    continue

                try:
                    df = pd.read_csv(model_file, sep="\t")
                except Exception as e:
                    print(f"⚠️ Could not read '{model_file}': {e}")
                    continue

                print(f"\n📂 Processing {model_file}")
                print(f"📄 Columns: {df.columns.tolist()}")

                data = []

                # Base column (Query vs Passage)
                base_col = SIMILARITY_TYPES["Query vs Passage"].format(model.replace("sim_", ""))
                base_scores = df[base_col].dropna().clip(lower=0).tolist() if base_col in df.columns else []

                # Compute normality for base distribution
                if base_scores:
                    base_normality = compute_normality_metrics(base_scores)
                else:
                    base_normality = {"Skewness": np.nan, "Kurtosis": np.nan, "Jarque-Bera": np.nan}

                # --- Iterate over similarity types in desired order ---
                for sim_type_label in SIMILARITY_ORDER:
                    col_template = SIMILARITY_TYPES[sim_type_label]
                    col_name = col_template.format(model.replace("sim_", ""))
                    if col_name in df.columns:
                        scores = df[col_name].dropna().clip(lower=0).tolist()
                        print(f"✔️ {col_name} – samples: {len(scores)}")
                        data.extend([(sim_type_label, score) for score in scores])

                        if sim_type_label != "Query vs Passage" and base_scores:
                            ovl = compute_distribution_overlap(base_scores, scores, "ovl")
                            jsd = compute_distribution_overlap(base_scores, scores, "jsd")
                            wd = compute_distribution_overlap(base_scores, scores, "wasserstein")
                            normality = compute_normality_metrics(scores)

                            overlap_results.append({
                                "Dataset": dataset_name,
                                "LLM": llm_name,
                                "File": file_base,
                                "Model": model,
                                "Compared Query Type": sim_type_label,
                                "OVL": round(ovl, 4),
                                "JSD": round(jsd, 4),
                                "Wasserstein": round(wd, 4),
                                "Skewness": normality["Skewness"],
                                "Kurtosis": normality["Kurtosis"],
                                "Jarque-Bera": normality["Jarque-Bera"],
                                "Base_Skewness": base_normality["Skewness"],
                                "Base_Kurtosis": base_normality["Kurtosis"],
                                "Base_Jarque-Bera": base_normality["Jarque-Bera"]
                            })
                    else:
                        print(f"⚠️ Missing column: {col_name}")

                if not data:
                    continue

                # --- Plot distributions ---
                df_plot = pd.DataFrame(data, columns=["Query Type", "Similarity"])
                df_plot["Query Type"] = pd.Categorical(
                    df_plot["Query Type"], 
                    categories=SIMILARITY_ORDER, 
                    ordered=True
                )

                plt.figure(figsize=(10, 6))
                ax = sns.kdeplot(
                    data=df_plot,
                    x="Similarity",
                    hue="Query Type",
                    fill=True,
                    common_norm=False,
                    alpha=0.4,
                    palette=QUERY_COLORS
                )

                # Title and labels
                plt.title(file_base, fontsize=18)
                plt.xlabel("Cosine Similarity", fontsize=14)
                plt.ylabel("Density", fontsize=14)
                plt.xlim(0, 1)
                plt.ylim(0, 4)   # ✅ fixed max height
                plt.grid(True, linestyle="--", alpha=0.6)

                # --- Explicit legend ---
                handles = [
                    mpatches.Patch(color=QUERY_COLORS[qtype], label=qtype)
                    for qtype in SIMILARITY_ORDER if qtype in df_plot["Query Type"].unique()
                ]
                ax.legend(handles=handles, title="Query Type", fontsize=13, title_fontsize=14)

                # Save the plot
                output_subdir = OUTPUT_DIR / dataset_name / llm_name / model
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_file = output_subdir / f"{file_base}.png"
                plt.tight_layout()
                plt.savefig(output_file, dpi=300)
                plt.close()
                print(f"✅ Saved plot: {output_file}")

# Save all metrics including base normality
if overlap_results:
    df_overlap = pd.DataFrame(overlap_results)
    overlap_file = OUTPUT_DIR / "overlap_results_with_base_normality.csv"
    df_overlap.to_csv(overlap_file, index=False)
    print(f"\n📝 All metrics saved in: {overlap_file}")
else:
    print("\n⚠️ No metrics computed.")
