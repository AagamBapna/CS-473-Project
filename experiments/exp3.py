"""Experiment 2 (KNN curve, random selection) + Experiment 3 (K-Means++ selection, heatmap, curves)."""
from exp_common import *
import json, time
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import approach1
P1 = knn_params(1)
SIZES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
df = load_data(); T, P = split_TP(df, seed=0); T = T.reset_index(drop=True)
# Same text representation as Approach 1 (its tokenizer: spelling + stemming, unigram TF-IDF), fitted on all of T (text only)
tok = approach1.Tokenizer(P1).fit(T.text.tolist())
vec = TfidfVectorizer(tokenizer=tok, preprocessor=lambda s: s, token_pattern=None, lowercase=False, sublinear_tf=True, norm="l2")
X = vec.fit_transform(T.text.tolist())
qidx = {q: np.where(T.q.values == q)[0] for q in T.q.unique()}

def evaluate(mask, tag):
    pred = a1_fit_predict(T[mask], P, **P1)
    m, per = per_question_qwk(P.y.values, pred, P.q.values)
    return dict(mean_per_q_qwk=m, pooled_qwk=qwk(np.rint(P.y.values * 2).astype(int), np.rint(pred * 2).astype(int), 11),
                large_q=np.mean([v for k, v in per.items() if k <= 10]), small_q=np.mean([v for k, v in per.items() if k > 10]), n_train=int(mask.sum()))

def cluster_select(N, size, seed):
    """Per question: K-Means++ with min(N, budget) clusters on T's TF-IDF rows; pick answers
    round-robin over clusters, nearest to each centroid first, until the budget is used."""
    mask = np.zeros(len(T), bool)
    for q, idx in qidx.items():
        b = max(1, int(round(size * len(idx))))
        k = max(1, min(N, b, len(idx)))
        km = KMeans(n_clusters=k, init="k-means++", n_init=3, random_state=seed).fit(X[idx])
        d = km.transform(X[idx]); lab = km.labels_
        queues = [list(np.array([i for i in np.argsort(d[:, c]) if lab[i] == c])) for c in range(k)]
        chosen = []
        while len(chosen) < b and any(queues):
            for qu in queues:
                if qu and len(chosen) < b:
                    chosen.append(qu.pop(0))
        mask[idx[chosen]] = True
    return mask

path = os.path.join(RES, "exp3_results.csv")
out = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
done = {(r["experiment"], int(r["N"]), int(r["seed"]), round(r["size"], 2)) for r in out}
t0 = time.time()
# ---- Experiment 2 (KNN, random nested subsets), 5 seeds
for seed in range(5):
    for s, mask in nested_subsets(T, seed, SIZES).items():
        if ("exp2_random", 0, seed, round(s, 2)) in done:
            continue
        out.append(dict(experiment="exp2_random", N=0, seed=seed, size=s, **evaluate(mask, "r")))
    print(f"random seed {seed} done [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(out).to_csv(os.path.join(RES, "exp3_results.csv"), index=False)
# ---- Experiment 3 heatmap: N x size (1 seed), then 5 seeds at the N that is best for small sizes
def run_cluster(N, seed):
    for s in SIZES:
        if ("exp3_cluster", N, seed, round(s, 2)) in done:
            continue
        out.append(dict(experiment="exp3_cluster", N=N, seed=seed, size=s, **evaluate(cluster_select(N, s, seed), "c")))
        pd.DataFrame(out).to_csv(path, index=False)
    print(f"N={N} seed {seed} done [{time.time()-t0:.0f}s]", flush=True)
for N in (2, 3, 5, 8, 12, 20):
    run_cluster(N, 0)
cl = pd.DataFrame(out); cl = cl[(cl.experiment == "exp3_cluster") & (cl["size"] <= 0.3)]
bestN = int(cl.groupby("N").mean_per_q_qwk.mean().idxmax()); print("chosen N =", bestN, flush=True)
for seed in range(1, 5):
    run_cluster(bestN, seed)
