import pandas as pd
import random

# Path to your TSV file
input_file = "test_text.tsv"
output_file = "sampled_data.tsv"

# Number of samples to take
num_samples = 20

# Read the TSV file into a DataFrame
df = pd.read_csv(input_file, sep="\t")

# Ensure we don't sample more than available rows
num_samples = min(num_samples, len(df))

# Randomly sample 20 rows
sampled_df = df.sample(n=num_samples, random_state=42)

# Save sampled data to a new TSV file
sampled_df.to_csv(output_file, sep="\t", index=False)

print(f"Sampled {num_samples} rows saved to {output_file}")
