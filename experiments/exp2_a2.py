"""Experiment 2, Approach 2 (MiniLM embedding KNN) learning curve, for completeness next to Approaches 1 and 3.

  python exp2_a2.py            # 5 random orderings of T -> results/exp2_approach2.csv

Same T/P split and nested subsets as exp2_a3.py / exp3.py. The MiniLM encoder is a fixed pretrained model, so the
precomputed embeddings (weights/approach2/train_embeddings.npz) carry no label information; the KNN index, question
centring and neighbour grades come from the training subset only.
"""
from exp_common import *
import time
P2 = knn_params(2)
path = os.path.join(RES, "exp2_approach2.csv")
df = load_data(); T, P = split_TP(df, seed=0)
print(f"T={len(T)} P={len(P)} rows; Approach 2 settings {P2}", flush=True)
out = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
done = {(int(r["seed"]), round(float(r["size"]), 2)) for r in out}
for seed in range(5):
    for s, mask in nested_subsets(T, seed).items():
        if (seed, round(s, 2)) in done:
            continue
        t0 = time.time(); Ts = T[mask]
        pred = a2_fit_predict(Ts, P, **P2)
        m, per = per_question_qwk(P.y.values, pred, P.q.values)
        pooled = qwk(np.rint(P.y.values * 2).astype(int), np.rint(pred * 2).astype(int), 11)
        big = np.mean([v for k, v in per.items() if k <= 10]); small = np.mean([v for k, v in per.items() if k > 10])
        out.append(dict(approach="Approach 2: MiniLM KNN", seed=seed, size=s, n_train=len(Ts), mean_per_q_qwk=m, pooled_qwk=pooled, large_q=big, small_q=small))
        print(f"seed {seed} size {s:.0%} n={len(Ts)}: per-q {m:.4f} (large {big:.3f}, small {small:.3f}) pooled {pooled:.4f} [{time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(out).sort_values(["seed", "size"]).to_csv(path, index=False)
print("done", flush=True)
