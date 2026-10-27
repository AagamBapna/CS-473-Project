# CS-473-Project
## Approach 1: KNN with TF-IDF

| path | purpose |
|---|---|
| `submission/approach1.py` | grader entrypoint (`load_model` / `predict`), inference only; also exposes `fit(train_rows, **params)` for CV and Experiments 2-3 |
| `submission/weights/approach1/` | `model.npz` (idf, training TF-IDF rows, grades, question ids), `vocab.json`, `config.json` (tuned settings + CV result) |

Porter stemming + edit-distance spelling normalisation, unigrams, sublinear tf, K=5 same-question
neighbours, similarity-weighted mean with offset -0.1. 5-fold CV (per-question QWK averaged over
questions 1-10): 0.493 ± 0.016.
