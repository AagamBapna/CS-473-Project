# Experiments 1-3 (CS 47300 Part 1)

Everything here was run on `training_dataset_updated.csv` (19,225 answers, 91 questions) with the team's actual code:
`team/approach1.py`, `team/approach2.py` (unchanged copies) and Approach 3 (`../approach3`, `../submission`).
Metric everywhere: QWK computed per question and averaged (the leaderboard's definition); questions whose
true grades in the evaluation set are all identical are skipped (QWK undefined), as the grader does.

Run order: `python exp1.py`, `python exp2_a3.py`, `python exp3.py`, `python plots.py`  (set the paths at the
top of `exp_common.py`). `shim/` holds a stand-in Porter stemmer used only because NLTK was not installable on
the experiment machine; on Gradescope / `create_environment.py` the real NLTK is used.

## Experiment 1: comparison of the three approaches (same 5 folds, `folds_v2.csv`)
| Approach | Best settings | CV mean per-question QWK | Leaderboard |
|---|---|---|---|
| 1: TF-IDF KNN | K=5, stemming + spelling normalisation, no stop-word removal, similarity-weighted mean, offset -0.1 | 0.337 ± 0.053 | 0.406 |
| 2: MiniLM KNN (all-MiniLM-L6-v2) | K=11 vote (large questions), K=3 weighted mean (small questions), question-centred | 0.393 ± 0.042 | 0.409 |
| 3: stacked grader, no LLM | per-question stack (TF-IDF ridge + logistic + mined rubric + exemplar similarity) for questions 1-10; cross-question ridge on relative similarity features (char, word, LSA) for 11-91; per-question QWK cut points | 0.565 ± 0.010 | 0.496 |
| 3: + gpt-oss:120b few-shot grader + expected-QWK decoder | one LLM call per small question with all its graded answers as examples; blend 0.7; expected-QWK decoding | **0.699 ± 0.011** | 0.624 (before decoder) |
Per group: questions 1-10 -> A1 0.486, A2 0.524, A3 0.689; questions 11-91 -> A1 0.318, A2 0.377, A3 0.549 (0.700 with LLM).
Files: `results/exp1_table.csv`, `results/exp1_oof_a1_a2.csv`.

## Experiment 2: QWK vs training-set size (`figures/exp2_learning_curve.png`)
Split: T = 90 %, P = 10 % (proportional per question and grade). Nested subsets of T at 10 %...90 %
(every subset contains the previous one and covers every question and grade). Approach 3 (best approach,
without the LLM stage): 3 random orderings; TF-IDF KNN: 5. Mean per-question QWK on P:

| training answers | 1.7k (10%) | 3.5k | 5.2k | 6.9k | 8.7k | 10.4k | 12.1k | 13.8k | 15.6k (90%) |
|---|---|---|---|---|---|---|---|---|---|
| Approach 3 | 0.076 | 0.135 | 0.255 | 0.367 | 0.415 | 0.428 | 0.422 | 0.439 | 0.449 |
| TF-IDF KNN | 0.128 | 0.193 | 0.262 | 0.268 | 0.306 | 0.341 | 0.350 | 0.358 | 0.366 |

Findings: QWK rises steeply and then levels off, as hypothesised. Questions 1-10 (hundreds of answers even at
10 %) are near their plateau early (0.54 -> 0.67); questions 11-91 drive the curve because 10 % of T is only
~2 graded answers per question (QWK ~0) and they keep gaining up to 90 %. Below ~5k answers plain KNN beats the
trained model (a learned model needs a minimum of data; KNN only needs one similar example), above it the
trained model wins by a growing margin. Note the LLM stage was excluded here (it would need ~700 API calls);
few-shot LLM grading is the one method known to work well in exactly this small-data regime.

## Experiment 3: K-Means++ selection of the graded set (`figures/exp3_heatmap.png`, `figures/exp3_curves.png`)
Same T/P split and sizes; same text representation as Approach 1 (its tokenizer + TF-IDF). Per question, T's
answers are clustered with K-Means++ into N clusters (N capped at the budget); the graded subset is chosen
round-robin across clusters, nearest-to-centroid first. TF-IDF KNN (Approach 1 settings) is then trained on
that subset and evaluated on P. Heatmap: N in {2,3,5,8,12,20} x 10 sizes (seed 0). N = 8 was chosen because it
has the best mean QWK over the small sizes (<= 30 %), which is where the hypothesis applies; curves use 5 seeds.

| size | 5% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.036 | 0.128 | 0.193 | 0.262 | 0.268 | 0.306 | 0.341 | 0.350 | 0.358 | 0.366 |
| clustered (N=8) | 0.039 | 0.203 | 0.201 | 0.272 | 0.324 | 0.323 | 0.327 | 0.355 | 0.344 | 0.355 |

Findings: clustering helps when the graded set is small (+0.075 at 10 %, +0.06 at 40 %; gains at every size
up to 50 %), supporting the hypothesis: picking one answer per cluster gives each new answer a more similar
graded neighbour. From 60 % on the two curves converge and clustering is slightly worse (-0.01): when nearly
every answer is graded anyway, the "nearest-to-centroid-first" rule leaves out exactly the atypical answers,
which are the ones KNN most needs an example of. On the ASAP questions (1-10) clustering hurts at every size
beyond 10 % (typical answers are over-represented in the graded set), on the CS questions it helps up to ~55 %.
Files: `results/exp3_results.csv`, `results/exp3_curve_table.csv`.
