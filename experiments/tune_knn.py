"""Tuning tables for Approaches 1 and 2 on the shared folds (handout: K, stop words, stemming, n-grams, combination rule; embedding K and rule)."""
from exp_common import *
import json, time
df = load_data(); fold = make_folds(df)
def cv(fn, **params):
    pred = np.zeros(len(df))
    for f in range(5):
        pred[fold == f] = fn(df[fold != f], df[fold == f], **params)
    m, per = per_question_qwk(df.y.values, pred, df.q.values)
    return m, np.mean([v for k, v in per.items() if k <= 10]), np.mean([v for k, v in per.items() if k > 10])
rows = []
base = dict(stopwords=False, stem=True, spell=True, ngram_max=1, sublinear_tf=True, min_df=1, k=5, weighting="similarity", aggregate="mean", same_question=True, offset=-0.1)
grid = [("K=3", dict(k=3)), ("K=5 (chosen)", {}), ("K=7", dict(k=7)), ("K=11", dict(k=11)), ("K=21", dict(k=21)),
        ("stop-word removal on", dict(stopwords=True)), ("no stemming", dict(stem=False)), ("no spelling normalisation", dict(spell=False)),
        ("word 1-2-grams", dict(ngram_max=2)), ("raw tf (not sublinear)", dict(sublinear_tf=False)),
        ("unweighted mean", dict(weighting="uniform")), ("weighted vote", dict(aggregate="vote")), ("no offset", dict(offset=0.0))]
t0 = time.time()
for name, over in grid:
    m, big, small = cv(a1_fit_predict, **{**base, **over}); rows.append(dict(approach="1", setting=name, qwk=m, large=big, small=small))
    print(f"A1 {name:28s} {m:.4f} (large {big:.3f}, small {small:.3f}) [{time.time()-t0:.0f}s]", flush=True)
import approach2
b2 = dict(approach2.DEFAULT_PARAMS)
grid2 = [("K large=11 vote, K small=3 mean (chosen)", {}), ("K large=5", dict(k_large=5)), ("K large=21", dict(k_large=21)), ("large: weighted mean", dict(combine_large="wmean")),
         ("K small=1", dict(k_small=1)), ("K small=5", dict(k_small=5)), ("K small=7", dict(k_small=7)), ("small: vote", dict(combine_small="vote")), ("no question centring", dict(center=False))]
for name, over in grid2:
    m, big, small = cv(a2_fit_predict, **{**b2, **over}); rows.append(dict(approach="2", setting=name, qwk=m, large=big, small=small))
    print(f"A2 {name:40s} {m:.4f} (large {big:.3f}, small {small:.3f})", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "tuning_a1_a2.csv"), index=False)
