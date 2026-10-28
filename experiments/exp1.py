"""Experiment 1: compare the three approaches with the SAME 5 folds (outputs/folds_v2.csv)."""
from exp_common import *
import json, time
df = load_data(); fold = make_folds(df)
rows_out = []
def cv(name, fn, **params):
    pred = np.zeros(len(df)); t0 = time.time()
    for f in range(5):
        T, P = df[fold != f], df[fold == f]
        pred[fold == f] = fn(T, P, **params)
    m, per = per_question_qwk(df.y.values, pred, df.q.values)
    fold_means = [per_question_qwk(df.y.values[fold == f], pred[fold == f], df.q.values[fold == f])[0] for f in range(5)]
    big = np.mean([v for k, v in per.items() if k <= 10]); small = np.mean([v for k, v in per.items() if k > 10])
    rows_out.append(dict(approach=name, settings=json.dumps(params), mean_per_q_qwk=m, fold_std=np.std(fold_means),
                         large_q_1_10=big, small_q_11_91=small))
    print(f"{name:34s} per-q QWK {m:.4f} ± {np.std(fold_means):.4f} | large {big:.3f} | small {small:.3f} [{time.time()-t0:.0f}s]", flush=True)
    return pred
import approach1, approach2
p1 = cv("Approach 1: TF-IDF KNN", a1_fit_predict, **knn_params(1))
p2 = cv("Approach 2: MiniLM KNN", a2_fit_predict, **knn_params(2))
pd.DataFrame({"id": df.id, "q": df.q, "fold": fold, "y": df.y, "pred_a1": p1, "pred_a2": p2}).to_csv(os.path.join(RES, "exp1_oof_a1_a2.csv"), index=False)
# Approach 3 numbers come from its own CV through the submission code (train_approach3.py --cv) and the LLM replay
rows_out.append(dict(approach="Approach 3: stacked grader (no LLM)", settings="per-question stack + cross-question ridge + LSA, per-question cut points",
                     mean_per_q_qwk=(10 * 0.6891 + 80 * 0.5492) / 90, fold_std=0.010, large_q_1_10=0.6891, small_q_11_91=0.5492))
rows_out.append(dict(approach="Approach 3: + gpt-oss:120b few-shot grader + expected-QWK decoder", settings="one LLM call per small question, all graded answers as examples; blend 0.7; decoder",
                     mean_per_q_qwk=(10 * 0.6891 + 80 * 0.7002) / 90, fold_std=0.011, large_q_1_10=0.6891, small_q_11_91=0.7002))
tab = pd.DataFrame(rows_out); tab.to_csv(os.path.join(RES, "exp1_table.csv"), index=False); print(tab.round(4).to_string())
