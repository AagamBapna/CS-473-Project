"""Experiment 2, Approach 3 learning curve (the best approach): QWK on P vs. size of the graded set.

  python exp2_a3.py                 # CPU model only (no GenAI stage), seeds 0 1 2   -> results/exp2_approach3.csv
  python exp2_a3.py --llm --seeds 0 # submitted model incl. the gpt-oss:120b stage   -> results/exp2_approach3_llm.csv
                                    # (needs CS473_CRED_DIR/credentials.py; ~81 API calls per size)

Resumable: finished (seed, size) rows are skipped. Approach 1's curve is produced by exp3.py (exp2_random rows).
"""
from exp_common import *
import argparse, time
ap = argparse.ArgumentParser()
ap.add_argument("--llm", action="store_true", help="also run the GenAI few-shot stage on the small questions")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="random orderings of T")
ap.add_argument("--n-jobs", type=int, default=2)
args = ap.parse_args()

label = "Approach 3 (+LLM)" if args.llm else "Approach 3 (no LLM)"
path = os.path.join(RES, "exp2_approach3_llm.csv" if args.llm else "exp2_approach3.csv")
df = load_data(); T, P = split_TP(df, seed=0)
print(f"T={len(T)} P={len(P)} rows; {label}; seeds {args.seeds}", flush=True)
out = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
done = {(int(r["seed"]), round(float(r["size"]), 2)) for r in out}
for seed in args.seeds:
    subs = nested_subsets(T, seed)
    for s, mask in subs.items():
        if (seed, round(s, 2)) in done:
            continue
        t0 = time.time(); Ts = T[mask]
        pred = a3_fit_predict(Ts, P, n_jobs=args.n_jobs, llm=args.llm)
        m, per = per_question_qwk(P.y.values, pred, P.q.values)
        pooled = qwk(np.rint(P.y.values * 2).astype(int), np.rint(pred * 2).astype(int), 11)
        big = np.mean([v for k, v in per.items() if k <= 10]); small = np.mean([v for k, v in per.items() if k > 10])
        out.append(dict(approach=label, seed=seed, size=s, n_train=len(Ts), mean_per_q_qwk=m, pooled_qwk=pooled, large_q=big, small_q=small))
        print(f"seed {seed} size {s:.0%} n={len(Ts)}: per-q {m:.4f} (large {big:.3f}, small {small:.3f}) pooled {pooled:.4f} [{time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(out).sort_values(["seed", "size"]).to_csv(path, index=False)
print("done", flush=True)
