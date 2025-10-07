import pandas as pd

# Load the input TSV
df = pd.read_csv("test_text_similarities_3models.tsv", sep="\t")

# Select and rename the relevant columns
output_df = df[[
    "query",
    "passage",
    "sim_query_vs_passage_all-MiniLM-L6-v2",
    "sim_query_vs_passage_msmarco-distilbert-base-v4",
    "sim_query_vs_passage_paraphrase-MiniLM-L12-v2"
]].rename(columns={
    "sim_query_vs_passage_all-MiniLM-L6-v2": "similarity_all-MiniLM-L6-v2",
    "sim_query_vs_passage_msmarco-distilbert-base-v4": "similarity_all-mpnet-base-v2",
    "sim_query_vs_passage_paraphrase-MiniLM-L12-v2": "similarity_paraphrase-MiniLM-L6-v2"
})

# Save to new TSV
output_df.to_csv("test_text_similarities.tsv", sep="\t", index=False)
print("TSV file generated successfully!")
