"""Experiment 2, Approach 3 (CPU model) learning curve: slow part, run in background."""
from exp_common import *
import time
df = load_data(); T, P = split_TP(df, seed=0)
path = os.path.join(RES, "exp2_approach3.csv")
out = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
done = {(r["seed"], round(r["size"], 2)) for r in out}
for seed in (0, 1, 2):
    subs = nested_subsets(T, seed)
    for s, mask in subs.items():
        if (seed, round(s, 2)) in done:
            continue
        t0 = time.time(); Ts = T[mask]
        pred = a3_fit_predict(Ts, P)
        m, per = per_question_qwk(P.y.values, pred, P.q.values)
        pooled = qwk(np.rint(P.y.values * 2).astype(int), np.rint(pred * 2).astype(int), 11)
        big = np.mean([v for k, v in per.items() if k <= 10]); small = np.mean([v for k, v in per.items() if k > 10])
        out.append(dict(approach="Approach 3 (no LLM)", seed=seed, size=s, n_train=len(Ts), mean_per_q_qwk=m, pooled_qwk=pooled, large_q=big, small_q=small))
        print(f"seed {seed} size {s:.0%} n={len(Ts)}: per-q {m:.4f} (large {big:.3f}, small {small:.3f}) pooled {pooled:.4f} [{time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(out).to_csv(os.path.join(RES, "exp2_approach3.csv"), index=False)
