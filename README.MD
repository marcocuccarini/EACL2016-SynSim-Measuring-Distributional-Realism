# Project README

## 📌 Ordine di esecuzione degli script

Per far funzionare correttamente l’intero progetto, **gli script devono essere eseguiti in sequenza**, perché ogni step dipende dall’output dello step precedente.

| Step | Script                        | Descrizione                                                                     |
| ---- | ----------------------------- | ------------------------------------------------------------------------------- |
| 1    | `2`    | Genera domande single-passage (`inpair`, `base`, `motivation`).                 |
| 2    | `2.1` 	| Genera domande dual-passage per maggiore coerenza.                              |
| 3    | `3`    | Calcola la similarità query → passage con modelli di embedding.                 |
| 4    | `4`    | Analizza e visualizza le distribuzioni delle similarità.                        |
| 5    | `5`    | Valuta ogni domanda secondo cinque metriche e salva in JSON.                    |
| 6    | `6`    | Aggrega media e varianza delle metriche e salva risultati finali in CSV e JSON. |

### ⚠️ Nota importante



* **Resume automatico**: gli script possono riprendere dal punto interrotto se un file esiste già, evitando di ricominciare da zero.

## 📂 Struttura delle cartelle

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

## ✅ Suggerimenti

* Eseguire gli script uno alla volta nell'ordine indicato.
* Verificare che i dataset siano scaricati e decompressi correttamente.
* Usare ambienti virtuali con tutte le dipendenze installate.
* Controllare i file di output parziali per confermare l’avanzamento del processo.
