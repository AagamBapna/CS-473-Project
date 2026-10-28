# Approach 3 — Rubric-Distilled, Rater-Aware Stacked Grader

**One-line idea:** instead of copying grades from similar answers (Approaches 1 and 2), learn *what earns points*. The pipeline:
1. Discover each question's rubric automatically (mined phrases, optionally an LLM-written rubric).
2. Look at every answer through several different models.
3. Combine those opinions with a learned stacker.
4. Convert the score to a grade with cut-points tuned for the exact metric being graded.

Nothing is specific to the 10 training questions. Score ranges, spelling dictionaries and rubrics are all learned from whatever graded data the pipeline receives, so it retrains on new questions unchanged.

```
                 ┌─ TF-IDF ridge (word 1-2g on spell-fixed text + char 2-5g) ─┐
 student answer ─┼─ TF-IDF logistic regression (class probabilities) ─────────┤
 (spell-fixed    ├─ Mined rubric: credit / misconception phrases ─────────────┤   level-2 stacker
  per question)  ├─ Exemplar similarity to best answers at each grade ────────┼─► ridge + gradient ─► cut-points ─► grade
                 ├─ Surface stats (length, OOV rate, ...) ────────────────────┤   boosting blend      tuned for QWK
                 ├─ [Colab] sentence embeddings (mpnet, bge) ─────────────────┤   (per question)
                 ├─ [Colab] LLM-rubric coverage (Purdue GenAI Studio) ───────┤
                 └─ [Colab] question-conditioned DeBERTa-v3 regressor ───────┘
```

## Results so far (5-fold CV, Score1, CPU components only)

These are measured. The rows marked [Colab] in the diagram are not included yet.

| Configuration | Pooled QWK | Mean per-question QWK |
|---|---|---|
| 1. TF-IDF ridge, plain rounding | 0.738 ± 0.006 | 0.600 |
| 2. + tuned cut-points | 0.766 ± 0.002 | 0.639 |
| 3. + logistic regression | 0.777 ± 0.005 | 0.663 |
| 4. + mined rubric | 0.780 ± 0.003 | 0.658 |
| 5. + exemplar similarity | 0.783 ± 0.006 | 0.667 |
| 6. + surface stats | 0.785 ± 0.006 | 0.672 |
| 7. gradient-boosting stacker instead of ridge | 0.774 ± 0.007 | 0.655 |
| **8. blend stacker (FINAL, CPU only)** | **0.786 ± 0.006** | 0.673 |
| 9. final, plain rounding | 0.779 ± 0.003 | 0.664 |
| 10. final, cut-points tuned for per-question QWK (`--metric perq`) | 0.768 ± 0.009 | **0.689** |
| 11. mined rubric + surface stats only (no TF-IDF at all) | 0.740 ± 0.005 | 0.599 |
| final, trained on mean of both raters (`--target soft`) | 0.784 ± 0.004 | 0.662 |

The full table, including per-question columns, is in `outputs/ablation.csv`. For reference: plain TF-IDF KNN scored 0.634 pooled, and the two human raters agree with each other at 0.91.

**Which QWK is graded matters.** Tuning cut-points per question raises per-question QWK, but it *lowers* pooled QWK. Tuning all questions' cut-points jointly against pooled QWK gives the best pooled score. The default is `--metric pooled`. If the TA says grading uses the mean of per-question QWKs, run with `--metric perq`.

## How to run

**1. CPU part (any laptop, about 15 min the first time, then cached):**
```
python stack.py cv              # full model, 5-fold CV -> outputs/cv_results.csv, outputs/oof_approach3.csv
python stack.py cv --ablation   # ablation table for the report -> outputs/ablation.csv
python stack.py cv --target soft   # ablation: train on the mean of both raters instead of Score1
python stack.py rubric          # print/save the mined rubric for every question
python predict.py --selfcheck   # train on 80%, grade the other 20%, report QWK
```
This needs Python 3 with numpy, pandas, scipy and scikit-learn. `outputs/folds.csv` holds the shared folds. **Give this file to your teammates** so all three approaches use identical folds.

**2. GPU + LLM parts (Google Colab, free T4):** upload this folder to `My Drive/approach3` and open `approach3_colab.ipynb`.
- Part A embeds every answer with two sentence-embedding models. These are also exactly what the Approach 2 teammate needs.
- Part B calls Purdue GenAI Studio about 60 times to write a rubric per question per fold, then measures how well each answer covers it.
- Part C fine-tunes DeBERTa-v3-base, which can resume from the last finished fold.
- Part D re-runs the stack with everything. Every output lands in `outputs/`, and `stack.py` detects it automatically.

**3. Grade a new file:**
```
python predict.py --test path/to/test.csv --out outputs/predictions.csv
```
Column names are detected automatically (Kaggle style or the handout's names). To use the Colab signals for a test file, set `TEST_CSV` in the notebook and re-run Parts A–C first.

## Why each piece is there (and what to cite)

| Piece | What it does | Inspiration |
|---|---|---|
| Spelling normaliser | Fixes typos using each question's own frequent vocabulary (*difussion → diffusion*, *osmosious → osmosis*). Words common across the whole corpus are protected from "correction". This is the handout's edit-distance suggestion, done efficiently. | SymSpell-style delete matching |
| Mined rubric | For every 1–3-gram, a z-test of whether answers containing it score higher or lower. Keeps the top 40 non-redundant credit phrases and 20 misconception phrases. | Ramachandran et al. 2015, who replaced the 2012 Kaggle winner's hand-written regexes with mined patterns |
| Logistic + ridge on TF-IDF | Two differently shaped opinions on the same features | 2012 Kaggle winners (Tandalla) |
| Exemplar similarity | "How close is this to the best 3-point answers vs. the best 0-point answers?" | Bexte et al. 2022 (similarity-based scoring) |
| Score1 target (both-rater average tested) | Training on the mean of both raters was *slightly worse* than Score1 alone (last row of the table), probably because grading is against Score1. This is an honest negative result for the report. | Tandalla 2012 used both raters |
| Blend stacker | Ridge + gradient boosting over all opinions | Ormerod et al. 2022: ensembles of *different kinds* of models beat single models |
| Tuned cut-points | Turns the continuous score into grades, maximising QWK | Standard Kaggle recipe for QWK |
| [Colab] DeBERTa with rubric in the input | One model for all questions; the input is `question id + key ideas [SEP] answer` | In-context tuning (NAEP), Funayama et al. (key phrases as input) |
| [Colab] LLM rubric | The LLM writes the rubric; it doesn't grade. LLM-as-grader scores only about 0.61–0.68 on this dataset. | Kim et al. 2025 (AnalyticScore), Jiang & Bosch 2024 |

## How leakage is prevented

- **Inner out-of-fold features.** Every feature that uses labels is computed out-of-fold *inside* the training part (5 inner folds). The stacker therefore never sees a feature built from the answer's own label.
- **Cut-points** are learned from out-of-fold predictions on the training answers, never from the fold being scored.
- **LLM rubrics** for CV fold *f* are written only from fold *f*'s training answers. Mined rubrics for the DeBERTa input work the same way (`mined_rubric_folds.json`).
- **Label-free steps.** Spelling dictionaries and embeddings use unlabeled text only, which a real grader would also have.
- **One caveat** (standard stacking practice, worth stating in the report): the DeBERTa out-of-fold predictions fed to the stacker come from fold models that saw other folds' labels.

## Files

| File | Purpose |
|---|---|
| `common.py` | Loading (auto column names), shared folds, fast QWK, per-question and pooled cut-point tuning |
| `features.py` | Spelling normaliser and all CPU feature components |
| `stack.py` | Nested stacking engine, CV, ablation, rubric dump |
| `predict.py` | Train on all graded data and grade a new file; `--selfcheck` |
| `gpu_parts.py` | Embeddings, GenAI Studio rubric and coverage, DeBERTa training (used by the notebook) |
| `approach3_colab.ipynb` | Runs the GPU and LLM parts in order and resumes after disconnects |
| `outputs/mined_rubric.json` | Mined rubric per question (good report material) |

## Confirm on Ed

1. Is the graded QWK pooled over all answers or averaged per question? This sets `--metric`.
2. Is `Score1` the label?
3. The submission format for the code or model.

## Gradescope submission (autograder does not train)

The autograder only calls `load_model()` and `predict()`, so the model is trained here and saved:

```
python train_approach3.py --data data/train_updated.csv          # -> the repository root/weights/approach3 (~45 MB, ~95 s)
python train_approach3.py --data data/train_updated.csv --cv     # 5-fold CV through the exact submission code
```

**Two models, chosen by question size.** The updated dataset added 81 questions (11–91), each with about 25 graded answers on a 0–5 half-point scale.
- **Questions with at least 150 graded answers and whole-number grades** use the per-question stacked grader described above.
- **Smaller questions** use one *cross-question* model (`a3_small.py`). Each answer is described relative to its own question's graded answers:
  - similarity to the full-mark answers;
  - the grades of its nearest graded neighbours;
  - how much of the full-mark answers' shared vocabulary it uses;
  - its length compared with theirs.

  These are measured in three spaces: character n-grams, words, and **LSA** (a 100-dimensional "meaning" space from SVD, which links answers that use related words). One ridge model, trained on all small questions together with leave-one-out features, maps these to a grade. Predictions are stretched 1.5× around the question's mean grade, because QWK rewards using the full grade range; 1.5 was chosen by CV over three fold seeds.

**CV through the submission code** (new folds in `outputs/folds_v2.csv`, stratified by question × grade, robust to 25-answer questions). Scores are QWK per question, then averaged, as on the leaderboard:

| Question group | QWK |
|---|---|
| 10 large questions | 0.689 (fold-by-fold 0.689 ± 0.012) |
| 80 small questions (one skipped: every answer has the same grade) | 0.549 |
| **All 90, equal weight** | **0.565** |

For the small questions, plain per-question KNN (k=3) scored 0.32 in the same setting. Tuning steps on the small questions (5 folds × 3 fold seeds): similarity features 0.470 → +stretch 0.500 → +LSA 0.533. Tried and rejected: per-question calibration, a pairwise comparison model, boosting blends, other LSA sizes, self-trained word vectors. Details are in `small_q.py`.

**Version-proof.** The artifacts are plain `.npz` arrays and JSON. `a3_infer.py` and `a3_small.py` compute everything with numpy/scipy only, so the grader's library versions don't matter. The export step checks that the re-implemented features match the trained ones.

**Speed and robustness.**
- It loads in about 3 s and grades about 5,800 answers in about 16 s on CPU. It makes no GenAI calls.
- Grades are snapped to `questions[q]["allowed_grades"]` and returned as an int when whole, a float (e.g. 2.5) when not.
- Empty or missing text is handled.
- A question never seen in training gets a valid length-based fallback.
