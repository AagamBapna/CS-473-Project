# CS 47300 Project Part 1 — Short-Answer Grade Prediction (Team Salkians)

Leaderboard (hidden test set, mean per-question QWK): Approach 1 = 0.406, Approach 2 = 0.409, **Approach 3 = 0.624**.

## Contents
| Folder | What it is |
|---|---|
| `submission/` | The exact files graded on Gradescope: `approach1.py` (TF-IDF KNN), `approach2.py` (MiniLM KNN), `approach3.py` + `a3_infer.py`, `a3_small.py`, `a3_llm.py` (open approach: stacked grader + few-shot LLM), `genai_wrapper.py` (course wrapper), `weights/` (all trained artifacts, ~60 MB) |
| `approach3_training/` | Code that trains and exports Approach 3 (`train_approach3.py`, `stack.py`, `features.py`, `small_q.py`, `common.py`), the shared folds (`outputs/folds_v2.csv`), the ablation table, the mined rubrics, and `llm_eval/` (LLM evaluation script + held-out results); `evaluated_not_submitted/` holds the expected-QWK decoder discussed in the report |
| `experiments/` | Experiments 1–3 and the KNN tuning sweep: `exp1.py`, `exp2_a3.py`, `exp3.py`, `tune_knn.py`, `plots.py`, with `results/*.csv` and `figures/*.png` (the figures in the report) |

Weights are included (zip is well under 300 MB).

## 1. Environment
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r) for r in ['stopwords','punkt','punkt_tab','wordnet']]"
```
Or run the course's `create_environment.py --download-encoders` from `local_environment.zip`, which installs the same versions and the MiniLM encoder needed by Approach 2.

## 2. Run the graded models (inference only, no training)
From `submission/`, using the course checker (`local_test.py` and `questions.json` from `local_environment.zip`):
```
python local_test.py --submission-dir . --csv training_dataset_updated.csv --questions questions.json --assets local_assets/assets.json
```
Each approach prints `OK: <n> predictions`. Approach 3's LLM stage runs only when `credentials.py` holds a Purdue GenAI Studio key (copy `credentials_example.py` to `credentials.py`) and `--allow-api` is given; without a key it falls back to the trained model alone.

Programmatic use (what the grader calls):
```python
import approach3
model = approach3.load_model("weights", {}, {})
model.predict([{"id": "1", "question": "14", "text": "in main"}], {"14": {"allowed_grades": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]}})
```

## 3. Retrain Approach 3 (~90 s on CPU)
```
cd approach3_training
python train_approach3.py --data path/to/training_dataset_updated.csv --out ../submission/weights/approach3
python train_approach3.py --data path/to/training_dataset_updated.csv --cv        # 5-fold CV through the submission code (~7 min)
cd llm_eval && python llm_eval.py --folds 0 1 --csv path/to/training_dataset_updated.csv   # LLM evaluation (~160 API calls; needs credentials.py)
```
Approaches 1 and 2 are trained with the `fit()` function inside `approach1.py` / `approach2.py` and saved with `TfidfKNNModel.save()` / the embeddings file (see the docstrings at the top of each file).

## 4. Experiments 1–3 and tuning
```
cd experiments            # set the three paths at the top of exp_common.py first
python exp1.py            # Experiment 1: all approaches on the shared folds (~1 min)
python tune_knn.py        # KNN tuning tables (~6 min)
python exp2_a3.py         # Experiment 2, Approach 3 learning curve (~30 min)
python exp3.py            # Experiment 2 (KNN) + Experiment 3 (K-Means++ selection, heatmap, curves) (~20 min)
python plots.py           # figures -> experiments/figures/
```
All scripts save after each step and resume if interrupted. Results used in the report are already in `experiments/results/`.

## 5. Build the Gradescope zip
Zip the *contents* of `submission/` (no enclosing folder) with `credentials.py` containing a valid key. Never commit or share `credentials.py`.
