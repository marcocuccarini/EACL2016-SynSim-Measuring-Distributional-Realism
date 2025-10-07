# Project README

## 📌 Script Execution Order

To ensure the project works correctly, **the scripts must be executed sequentially**, as each step depends on the output of the previous one.

| Step | Script | Description |
| ---- | ------ | ----------- |
| 1    | `2`    | Generates single-passage questions (`inpair`, `base`, `motivation`). |
| 2    | `2.1`  | Generates dual-passage questions for better coherence. |
| 3    | `3`    | Calculates query → passage similarity using embedding models. |
| 4    | `4`    | Analyzes and visualizes similarity distributions. |
| 5    | `5`    | Evaluates each question according to five metrics and saves results in JSON. |
| 6    | `6`    | Aggregates mean and variance of metrics and saves final results in CSV and JSON. |

### ⚠️ Important Note

* **Automatic resume**: scripts can resume from the interrupted step if a file already exists, avoiding the need to start over from scratch.

## 📂 Folder Structure


```
datasets/
 └── <dataset_name>/
     ├── corpus.jsonl
     ├── queries.jsonl
     ├── qrels/
     │   ├── *_text.tsv
     │   ├── *_similarity.tsv
     │   └── <model_name>/
     │       ├── *_evaluated.json
     │       └── *_partial.json
     └── samples/
         └── sample_1.tsv
plots/
 └── <dataset_name>/
     └── <llm_name>/
         └── <model>/
             └── *.png
```

## ✅ Tips

* Run the scripts one at a time in the indicated order.  
* Ensure datasets are downloaded and decompressed correctly.  
* Use virtual environments with all dependencies installed.  
* Check partial output files to confirm progress.
