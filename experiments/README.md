# Experiments 1-3 (CS 47300 Part 1)

Everything here was run on `training_dataset_updated.csv` (19,225 answers, 91 questions) with the **submitted** code at the
repository root (`approach1.py`, `approach2.py`, `approach3.py` + `a3_*.py`); Approach 3 is retrained for every experiment
run with `training/train_approach3.py` and then graded through `approach3.py`, i.e. the exact grader path.
Metric everywhere: QWK computed per question and averaged (the leaderboard's definition); questions whose true grades in the
evaluation set are all identical are skipped (QWK undefined), as the grader does.

How to run (Python 3.12 with the grader's library versions, see `../README.md`):
```
export CS473_DATA=/path/to/training_dataset_updated.csv
python exp1.py                      # Experiment 1 (Approaches 1 and 2 on the shared folds)      -> results/exp1_*.csv
python exp2_a3.py                   # Experiment 2, Approach 3 without the GenAI stage, 3 orderings -> results/exp2_approach3.csv
CS473_CRED_DIR=/dir/with/credentials.py python exp2_a3.py --llm --seeds 0
                                    # Experiment 2, submitted Approach 3 incl. gpt-oss:120b stage   -> results/exp2_approach3_llm.csv
python exp3.py                      # Experiment 2 Approach 1 curve + Experiment 3 (K-Means++ selection) -> results/exp3_results.csv
python exp2_a2.py                   # Experiment 2, Approach 2 curve (not required; for completeness)    -> results/exp2_approach2.csv
python plots.py                     # figures/*.png, results/exp2_table.csv, results/exp3_curve_table.csv
```
All runs are resumable (finished rows are skipped). `shim/` holds a stand-in Porter stemmer that is used only if NLTK is not
installed; `results/exp3_results_shim_stemmer.csv` keeps the earlier Approach 1 runs made with it (they differ from the NLTK
runs below by at most 0.01 QWK at any size).

## Experiment 1: comparison of the three approaches (same 5 folds, `../training/outputs/folds_v2.csv`)
| Approach | Best settings | CV mean per-question QWK | Leaderboard |
|---|---|---|---|
| 1: TF-IDF KNN | K=5, stemming + spelling normalisation, no stop-word removal, similarity-weighted mean, offset -0.1 | 0.337 ± 0.053 | 0.406 |
| 2: MiniLM KNN (all-MiniLM-L6-v2) | K=11 vote (large questions), K=3 weighted mean (small questions), question-centred | 0.393 ± 0.042 | 0.409 |
| 3: stacked grader, no LLM | per-question stack (TF-IDF ridge + logistic + mined rubric + exemplar similarity) for questions 1-10; cross-question ridge on relative similarity features (char, word, LSA) for 11-91; per-question QWK cut points | 0.565 ± 0.010 | 0.496 |
| 3: + gpt-oss:120b few-shot grader + expected-QWK decoder | one LLM call per small question with all its graded answers as examples; blend 0.7; expected-QWK decoding | **0.699 ± 0.011** | 0.624 (before decoder) |
Per group: questions 1-10 -> A1 0.486, A2 0.524, A3 0.689; questions 11-91 -> A1 0.318, A2 0.377, A3 0.549 (0.700 with LLM).
Files: `results/exp1_table.csv`, `results/exp1_oof_a1_a2.csv`.

## Experiment 2: QWK vs training-set size (`figures/exp2_learning_curve.png`, `results/exp2_table.csv`)
Protocol (handout Section 4): the collection is split once into **T (90 %, 17,303 answers)** and **P (10 %, 1,922 answers)**;
within every question the answers are sorted by grade and every 10th goes to P, so P covers every question and grade level in
proportion. Training sizes are **10 %, 20 %, ... 90 % of T**, as nested subsets: each question's answers are put in a random order
that interleaves the grades, and subset *k* is the first *k*/10 of every question's order, so every subset contains the previous
one (the graded pile G grows) and covers every question and grade. At every size the approach is **retrained from scratch on
that subset only** (vocabularies, IDF, rubric, stacker, cut points, LLM example set) and predicts all of P. Repeated over random
orderings of T (Approach 3: 3, Approaches 1 and 2: 5); curves show mean ± std. The handout asks for the best approach
(Approach 3); the KNN curves are the Experiment 3 baseline (Approach 1) and, for completeness, Approach 2. Mean per-question
QWK on P:

| graded training answers | 1.7k (10%) | 3.5k | 5.2k | 6.9k | 8.7k | 10.4k | 12.1k | 13.8k | 15.6k (90%) |
|---|---|---|---|---|---|---|---|---|---|
| **Approach 3, submitted (+ gpt-oss:120b stage)**, 1 ordering | **0.332** | **0.530** | **0.626** | **0.628** | **0.659** | **0.649** | **0.681** | **0.676** | **0.700** |
| Approach 3, no LLM stage | 0.076 | 0.135 | 0.255 | 0.366 | 0.415 | 0.428 | 0.438 | 0.446 | 0.457 |
| Approach 1: TF-IDF KNN | 0.124 | 0.191 | 0.258 | 0.259 | 0.304 | 0.333 | 0.351 | 0.368 | 0.376 |
| Approach 2: MiniLM KNN (for completeness) | 0.145 | 0.244 | 0.305 | 0.325 | 0.323 | 0.337 | 0.359 | 0.375 | 0.377 |
| *questions 1-10 only:* Approach 3 no LLM / A1 / A2 | 0.54 / 0.34 / 0.43 | 0.61 / 0.40 / 0.47 | 0.62 / 0.42 / 0.48 | 0.63 / 0.45 / 0.49 | 0.64 / 0.46 / 0.49 | 0.66 / 0.47 / 0.50 | 0.67 / 0.47 / 0.50 | 0.67 / 0.48 / 0.50 | 0.67 / 0.47 / 0.49 |
| *questions 11-91 only:* Approach 3 no LLM / A1 / A2 | 0.01 / 0.09 / 0.10 | 0.07 / 0.16 / 0.21 | 0.20 / 0.23 / 0.28 | 0.33 / 0.23 / 0.30 | 0.38 / 0.28 / 0.30 | 0.39 / 0.31 / 0.31 | 0.40 / 0.33 / 0.34 | 0.41 / 0.35 / 0.36 | 0.43 / 0.36 / 0.36 |

Findings. The hypothesis holds: QWK rises steeply with the number of graded answers and then levels off. For questions 1-10
(hundreds of graded answers even at 10 %) both methods are close to their plateau early (Approach 3: 0.54 -> 0.67, flat from
~60 %; KNN: 0.34 -> 0.47, flat from ~50 %). The overall curve is driven by questions 11-91, where 10 % of T is only ~2 graded
answers per question (QWK ≈ 0) and every added answer still helps at 90 %. Below ~5k graded answers plain TF-IDF KNN beats the
trained grader (a learned model needs a minimum of data per question, KNN only needs one similar graded example); above it the
trained grader wins by a growing margin (+0.08 at 90 %). The two KNN approaches behave alike: MiniLM KNN is ahead of TF-IDF KNN
by 0.02-0.07 at small sizes (dense embeddings match paraphrases that share no words, which matters most when few graded answers
exist) and the two converge at 90 % (0.377 vs 0.376).
The submitted Approach 3 (with the gpt-oss:120b few-shot stage on questions 11-91; one random ordering, 81 API calls per size)
changes the shape of the curve: 0.33 with only 10 % of the collection graded, 0.63 at 30 %, and a plateau of 0.65-0.70 from
50 % on. On questions 11-91 it reaches 0.63 with ~4 graded answers per question (30 %), a level the trained model never
reaches and KNN reaches nowhere; on questions 1-10 the LLM stage is not used, so there the submitted model is the no-LLM model
(the small differences in that panel are the single ordering vs the mean of three). This is
the small-data regime the experiment is about: a few-shot LLM grader needs only a handful of graded examples to infer the
grading standard, whereas KNN and trained models need many per question. So the hypothesis holds for every method, but the
amount of grading needed before the tool becomes useful is far smaller with the LLM stage (10-20 % of the collection vs
50-60 % for the others).
Files: `results/exp2_approach3.csv`, `results/exp2_approach3_llm.csv`, `results/exp2_approach2.csv`, Approach 1 rows = `exp2_random`
in `results/exp3_results.csv`; summary `results/exp2_table.csv`.

## Experiment 3: K-Means++ selection of the graded set (`figures/exp3_heatmap.png`, `figures/exp3_curves.png`)
Same T/P split and sizes (plus 5 %); same text representation as Approach 1 (its tokenizer: spelling normalisation + Porter
stemming, unigram TF-IDF). Per question, T's answers are clustered with K-Means++ into N clusters (N capped at the budget); the
graded subset is chosen round-robin across clusters, nearest-to-centroid first. TF-IDF KNN (Approach 1 settings) is then trained
on that subset and evaluated on P. Heatmap: N in {2, 3, 5, 8, 12, 20} x 10 sizes (seed 0). **N = 12** was chosen because it has the
best mean QWK over the small sizes (<= 30 %), which is where the hypothesis applies (N = 5 and 20 are within 0.001); curves use
5 seeds.

| size | 5% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.036 | 0.124 | 0.191 | 0.258 | 0.259 | 0.304 | 0.333 | 0.351 | 0.368 | 0.376 |
| clustered (N=12) | 0.043 | 0.152 | 0.192 | 0.239 | 0.284 | 0.282 | 0.300 | 0.320 | 0.339 | 0.378 |

Findings: the effect of clustering is small (|Δ| <= 0.03 overall, comparable to the spread across seeds) and only partly in
the hypothesised direction. With the smallest graded sets it helps (+0.007 at 5 %, +0.028 at 10 %; on questions 1-10, where
clusters are well populated, +0.05 at 5 % and +0.04 at 10 %): picking one answer per cluster gives each new answer a more
similar graded neighbour. At 30 % and from 50 % to 80 % random selection is better (-0.02 to -0.03): when most answers are
graded anyway, the "nearest-to-centroid-first" rule leaves out exactly the atypical answers, which are the ones KNN most needs
an example of. At 90 % the two curves converge (0.378 vs 0.376), as expected when nearly the whole collection is graded. On
the ASAP questions (1-10) clustering helps only up to 10 % and hurts beyond (typical answers are over-represented in the graded
set); on the CS questions (11-91, ~22 answers each) the sign of the gain alternates with size, i.e. no reliable effect. So the
hypothesis is supported only for very small budgets (<= 10 % of the collection), and not beyond.
Files: `results/exp3_results.csv`, `results/exp3_curve_table.csv`.
