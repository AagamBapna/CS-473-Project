"""Approach 3 engine: Rubric-Distilled, Rater-Aware Stacked Grader.

For each question separately:
  1. Level-1 components (features.py) turn answers into a few numbers each.
     For TRAINING answers these numbers are out-of-fold (5 inner folds), so the stacker
     never sees a feature computed from the answer's own label.
  2. Optional external signals from Colab are joined by id:
       outputs/oof_tx_<name>.csv   id,pred      (transformer, normalised 0-1, out-of-fold)
       outputs/test_tx_<name>.csv  id,pred      (same model's predictions on a test file)
       outputs/emb_<tag>.npz       ids, X       (sentence embeddings, any rows)
       outputs/llm_feats.csv       id,context,<cols>   (LLM-rubric coverage features)
  3. A level-2 stacker (ridge / gradient boosting / blend) predicts a 0-1 score.
  4. Cut points that maximise QWK are tuned on the stacker's out-of-fold training
     predictions and applied to the new answers. By default all questions' cut points are
     tuned jointly for POOLED QWK (one kappa over every answer); --metric perq instead
     maximises each question's own QWK.

Usage:
  python stack.py cv                # 5-fold CV of the full model (+ saves OOF predictions)
  python stack.py cv --ablation     # ablation table for the report
  python stack.py cv --target soft  # train on the mean of both raters (ablation)
  python stack.py cv --metric perq  # tune cut points for mean per-question QWK instead of pooled
  python stack.py rubric            # dump mined rubrics (report + DeBERTa prompt)
"""
import argparse
import glob
import json
import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from common import (OUT, load, make_folds, qwk, fit_thresholds, fit_thresholds_pooled,
                    apply_thresholds, soft_target)
from features import (SpellNormalizer, global_counts, TfidfRidge, TfidfLogit, RubricMiner,
                      Exemplar, EmbRidge, Surface)

INNER = 5
SEED = 0
HERE_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ data preparation
def prepare(df):
    """Add row index and spell-normalised text. Spelling dictionaries use unlabeled text
    only (all answers to the question), which a real grader also has."""
    df = df.copy().reset_index(drop=True)
    df["row"] = np.arange(len(df))
    G = global_counts(df.text)
    spellers, norm = {}, np.empty(len(df), object)
    for q, d in df.groupby("q"):
        sp = SpellNormalizer(global_freq=G).fit(d.text)
        spellers[q] = sp
        norm[d.index] = [sp.normalize(s) for s in d.text]
    df["norm"] = norm
    return df, spellers


def load_external(df, mode="cv"):
    """Collect optional signals produced in Colab. Missing files are simply skipped."""
    ext = {"emb": {}, "cols": {}, "llm": None}
    for f in sorted(glob.glob(os.path.join(OUT, "emb_*.npz"))):
        z = np.load(f, allow_pickle=True)
        pos = {i: k for k, i in enumerate(z["ids"].tolist())}
        if all(i in pos for i in df.id):
            ext["emb"][os.path.basename(f)[4:-4]] = z["X"][[pos[i] for i in df.id]].astype(np.float32)
    for f in sorted(glob.glob(os.path.join(OUT, "oof_tx_*.csv"))):   # transformer OOF files
        tag = os.path.basename(f)[4:-4]                                    # e.g. tx_deberta
        oof = pd.read_csv(f)
        col = df.id.map(dict(zip(oof.id, oof.pred)))
        tf = os.path.join(OUT, f"test_{tag}.csv")
        if mode == "final" and os.path.exists(tf):
            t = pd.read_csv(tf)
            col = col.fillna(df.id.map(dict(zip(t.id, t.pred))))
        if col.notna().all():
            ext["cols"][tag] = col.values.astype(float)
    f = os.path.join(OUT, "llm_feats.csv")
    if os.path.exists(f):
        ext["llm"] = pd.read_csv(f)
    return ext


# ------------------------------------------------------------------ level 1
def make_components(blocks, ext, speller):
    comps = {}
    if "tfidf" in blocks:
        comps["tfidf"] = lambda: TfidfRidge(alpha=1.0)
    if "logit" in blocks:
        comps["logit"] = lambda: TfidfLogit(C=4.0)
    if "rubric" in blocks:
        comps["rubric"] = lambda: RubricMiner()
    if "exemplar" in blocks:
        comps["exemplar"] = lambda: Exemplar()
    if "surface" in blocks:
        comps["surface"] = lambda: Surface(speller)
    if "emb" in blocks:
        for tag, E in ext["emb"].items():
            comps[f"emb_ridge_{tag}"] = (lambda E=E, tag=tag: EmbRidge(E, tag))
            comps[f"emb_ex_{tag}"] = (lambda E=E, tag=tag: Exemplar(emb=E, tag=tag))
    return comps


def level1(tr, te, t, m, comps):
    """Returns {block: (F_tr_oof, F_te)}."""
    out = {}
    strat = tr.y.values if np.bincount(tr.y.values).min() >= INNER else None
    splitter = (StratifiedKFold(INNER, shuffle=True, random_state=SEED) if strat is not None
                else KFold(INNER, shuffle=True, random_state=SEED))
    splits = list(splitter.split(tr, strat if strat is not None else None))
    for name, make in comps.items():
        F_te = make().fit(tr, t, m).transform(te)
        F_tr = np.zeros((len(tr), F_te.shape[1]))
        for a, b in splits:
            F_tr[b] = make().fit(tr.iloc[a], t[a], m).transform(tr.iloc[b])
        out[name] = (F_tr, F_te)
    return out


def external_blocks(tr, te, ext, context):
    out = {}
    for tag, col in ext["cols"].items():
        out[tag] = (col[tr.row.values][:, None], col[te.row.values][:, None])
    if ext["llm"] is not None:
        L = ext["llm"]
        L = L[L.context.astype(str) == str(context)].set_index("id")
        cols = [c for c in L.columns if c != "context"]
        if len(L) and all(i in L.index for i in tr.id) and all(i in L.index for i in te.id):
            out["llm"] = (L.loc[tr.id, cols].values, L.loc[te.id, cols].values)
    return out


# ------------------------------------------------------------------ level 2
def make_stacker(kind):
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    hgb = lambda: HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
                                                min_samples_leaf=25, l2_regularization=1.0, random_state=SEED)
    if kind == "ridge":
        return ridge
    if kind == "hgb":
        return hgb
    if kind == "blend":
        class Blend:
            def fit(self, X, y):
                self.a, self.b = ridge().fit(X, y), hgb().fit(X, y)
                return self

            def predict(self, X):
                return 0.5 * self.a.predict(X) + 0.5 * self.b.predict(X)
        return Blend
    raise ValueError(kind)


def level2(F_tr, F_te, tr, m, stacker="blend", target="hard"):
    """Fit the stacker. Returns (out-of-fold predictions on the training answers,
    predictions on the new answers), both on the 0..m grade scale."""
    t = soft_target(tr) / m if target == "soft" else tr.y.values / m
    make = make_stacker(stacker)
    oof = np.zeros(len(tr))
    for a, b in KFold(INNER, shuffle=True, random_state=SEED + 1).split(F_tr):
        oof[b] = make().fit(F_tr[a], t[a]).predict(F_tr[b])
    return oof * m, make().fit(F_tr, t).predict(F_te) * m


def to_grades(P, mode="pooled"):
    """P: list of (oof_train, cont_new, y_train, m), one per question.
    mode 'round'  : plain rounding
         'perq'   : per-question cut points maximising each question's own QWK
         'pooled' : cut points tuned jointly to maximise QWK over all answers pooled
    Cut points are always learned from the TRAINING answers' out-of-fold predictions."""
    if mode == "round":
        return [np.clip(np.rint(c), 0, m).astype(int) for _, c, _, m in P]
    ths = [fit_thresholds(o, y, m) for o, _, y, m in P]
    if mode == "pooled":
        ths = fit_thresholds_pooled([o for o, *_ in P], [y for _, _, y, _ in P], [m for *_, m in P], init=ths)
    return [np.clip(apply_thresholds(c, th), 0, m) for (_, c, _, m), th in zip(P, ths)]


ALL_BLOCKS = ["tfidf", "logit", "rubric", "exemplar", "surface", "emb"]


def question_job(tr, te, ext, speller, context, blocks=ALL_BLOCKS, target="hard"):
    m = int(tr.y.max())
    if m == 0:
        return None
    t = (soft_target(tr) if target == "soft" else tr.y.values.astype(float)) / m
    L1 = level1(tr, te, t, m, make_components(blocks, ext, speller))
    L1.update(external_blocks(tr, te, ext, context))
    return m, L1


def assemble(L1, names):
    names = [n for n in L1 if any(n == b or n.startswith(b + "_") for b in names)]
    return (np.hstack([L1[n][0] for n in names]), np.hstack([L1[n][1] for n in names]), names)


# ------------------------------------------------------------------ public API
def fit_predict(train, test, stacker="blend", blocks=None, n_jobs=2, context="full", ext=None,
                target="hard", metric="pooled"):
    """Train on graded answers `train`, grade `test`. Works for any questions/score ranges.
    Questions in test with no training answers fall back to the most common grade overall."""
    allx = pd.concat([train.assign(_split="tr"), test.assign(_split="te")], ignore_index=True)
    allx, spellers = prepare(allx)
    ext = ext or load_external(allx, mode="final")
    print(f"external signals: embeddings={list(ext['emb'])} transformers={list(ext['cols'])} "
          f"llm={'yes' if ext['llm'] is not None else 'no'} | thresholds={metric}")
    tr_all, te_all = allx[allx._split == "tr"].copy(), allx[allx._split == "te"]
    tr_all["y"] = tr_all.y.astype(int)          # concat with unlabeled rows made it float
    if "y2" in tr_all and tr_all.y2.notna().all():
        tr_all["y2"] = tr_all.y2.astype(int)
    blocks = blocks or ALL_BLOCKS + list(ext["cols"]) + (["llm"] if ext["llm"] is not None else [])
    qs = [q for q in te_all.q.unique() if q in set(tr_all.q)]
    jobs = Parallel(n_jobs=n_jobs)(delayed(question_job)(tr_all[tr_all.q == q], te_all[te_all.q == q], ext,
                                                         spellers[q], context, target=target) for q in qs)
    pred = pd.Series(int(tr_all.y.mode()[0]), index=te_all.index)
    cont = pd.Series(np.nan, index=te_all.index)
    P, where = [], []
    for q, res in zip(qs, jobs):
        te = te_all[te_all.q == q]
        if res is None:            # every training answer has the same grade
            pred[te.index] = int(tr_all[tr_all.q == q].y.iloc[0])
            continue
        m, L1 = res
        F_tr, F_te, _ = assemble(L1, blocks)
        o, c = level2(F_tr, F_te, tr_all[tr_all.q == q], m, stacker, target)
        P.append((o, c, tr_all[tr_all.q == q].y.values, m))
        where.append(te.index)
    for idx, g, (_, c, _, _) in zip(where, to_grades(P, metric), P):
        pred[idx], cont[idx] = g, c
    return pd.DataFrame({"id": te_all.id.values, "q": te_all.q.values, "pred": pred.values.astype(int),
                         "cont": cont.values})


def _level1_cached(df, fold, ext, spellers, target, n_jobs):
    """Level-1 features are the slow part (~5 min on 2 CPUs); cache them so the ablation
    and repeated runs are fast. The cache key covers the code and every external input."""
    import hashlib
    import pickle
    sig = [target, len(df), int(df.id.sum()), int(fold.sum())]
    for f in [os.path.join(HERE_DIR, "features.py")] + \
            sorted(glob.glob(os.path.join(OUT, "emb_*.npz")) + glob.glob(os.path.join(OUT, "oof_tx_*.csv")) +
                   glob.glob(os.path.join(OUT, "llm_feats.csv"))):
        sig.append((os.path.basename(f), os.path.getmtime(f), os.path.getsize(f)))
    key = hashlib.md5(repr(sig).encode()).hexdigest()[:12]
    path = os.path.join(OUT, f".level1_cache_{key}.pkl")
    if os.path.exists(path):
        print("level-1 features loaded from cache")
        return pickle.load(open(path, "rb"))
    tasks = [(f, q) for f in range(fold.max() + 1) for q in sorted(df.q.unique())]
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(
        delayed(question_job)(df[(fold != f) & (df.q == q)], df[(fold == f) & (df.q == q)], ext, spellers[q], f,
                              target=target) for f, q in tasks)
    print(f"level-1 features done in {time.time() - t0:.0f}s")
    for old in glob.glob(os.path.join(OUT, ".level1_cache_*.pkl")):
        os.remove(old)
    pickle.dump((tasks, res), open(path, "wb"))
    return tasks, res


def run_cv(df, ablation=False, n_jobs=2, target="hard", metric="pooled"):
    df, spellers = prepare(df)
    fold = make_folds(df)
    ext = load_external(df, mode="cv")
    print(f"external signals: embeddings={list(ext['emb'])} transformers={list(ext['cols'])} "
          f"llm={'yes' if ext['llm'] is not None else 'no'} | target={target} | thresholds={metric}")
    tasks, res = _level1_cached(df, fold, ext, spellers, target, n_jobs)

    extra = list(ext["cols"]) + (["llm"] if ext["llm"] is not None else [])
    cpu = ["tfidf", "logit", "rubric", "exemplar", "surface"]
    full = cpu + (["emb"] if ext["emb"] else []) + extra
    configs = [("FINAL: full model, blend stacker", full, "blend", metric)]
    if ablation:
        configs = [
            ("1. TF-IDF ridge, plain rounding", ["tfidf"], "ridge", "round"),
            ("2. TF-IDF ridge + tuned thresholds", ["tfidf"], "ridge", metric),
            ("3. + logistic regression", ["tfidf", "logit"], "ridge", metric),
            ("4. + mined rubric", ["tfidf", "logit", "rubric"], "ridge", metric),
            ("5. + exemplar similarity", ["tfidf", "logit", "rubric", "exemplar"], "ridge", metric),
            ("6. + surface stats", cpu, "ridge", metric),
        ]
        k = 7
        if ext["emb"]:
            configs.append((f"{k}. + sentence embeddings", configs[-1][1] + ["emb"], "ridge", metric)); k += 1
        for e in extra:
            configs.append((f"{k}. + {e}", configs[-1][1] + [e], "ridge", metric)); k += 1
        configs += [
            (f"{k}. same, gradient-boosting stacker", full, "hgb", metric),
            (f"{k + 1}. FINAL: same, blend stacker", full, "blend", metric),
            (f"{k + 2}. final, plain rounding", full, "blend", "round"),
            (f"{k + 3}. final, per-question thresholds", full, "blend", "perq"),
            (f"{k + 4}. final, pooled thresholds", full, "blend", "pooled"),
            (f"{k + 5}. mined rubric + surface only", ["rubric", "surface"], "ridge", metric),
        ]
        seen, uniq = set(), []
        for c in configs:                       # drop exact repeats (e.g. pooled == default)
            key = (tuple(c[1]), c[2], c[3])
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        configs = [(f"{i + 1}. {n.split('. ', 1)[1]}",) + c[1:] for i, (n, *_), c in
                   zip(range(len(uniq)), uniq, uniq)]   # renumber after dropping repeats

    y = df.y.values
    rows, oof_final = [], None
    for name, blocks, stacker, mode in configs:
        pred = np.zeros(len(df), int)
        cont = np.zeros(len(df))
        for f in range(fold.max() + 1):
            P, where = [], []
            for (ff, q), r in zip(tasks, res):
                if ff != f or r is None:
                    continue
                m, L1 = r
                F_tr, F_te, _ = assemble(L1, blocks)
                tr = df[(fold != f) & (df.q == q)]
                o, c = level2(F_tr, F_te, tr, m, stacker, target)
                P.append((o, c, tr.y.values, m))
                where.append(df.index[(fold == f) & (df.q == q)])
            for idx, g, (_, c, _, _) in zip(where, to_grades(P, mode), P):
                pred[idx], cont[idx] = g, c
        fq = [qwk(y[fold == f], pred[fold == f]) for f in range(fold.max() + 1)]
        perq = {int(q): qwk(y[df.q == q], pred[df.q == q]) for q in sorted(df.q.unique())}
        perq_fold = [np.mean([qwk(y[(fold == f) & (df.q == q)], pred[(fold == f) & (df.q == q)])
                              for q in perq]) for f in range(fold.max() + 1)]
        rows.append({"config": name, "pooled_qwk": np.mean(fq), "pooled_std": np.std(fq),
                     "mean_per_q_qwk": np.mean(list(perq.values())), "per_q_std": np.std(perq_fold),
                     **{f"q{q}": v for q, v in perq.items()}})
        print(f"{name:42s} pooled {np.mean(fq):.4f} ±{np.std(fq):.4f} | per-question mean "
              f"{np.mean(list(perq.values())):.4f} ±{np.std(perq_fold):.4f}", flush=True)
        if "FINAL" in name:
            oof_final = pd.DataFrame({"id": df.id, "q": df.q, "fold": fold, "y": df.y, "pred": pred, "cont": cont})
    tab = pd.DataFrame(rows)
    suffix = ("_soft" if target == "soft" else "") + ("" if metric == "pooled" else f"_{metric}")
    tab.to_csv(os.path.join(OUT, f"{'ablation' if ablation else 'cv_results'}{suffix}.csv"), index=False)
    if oof_final is not None:
        oof_final.to_csv(os.path.join(OUT, f"oof_approach3{suffix}.csv"), index=False)
    return tab


def dump_rubrics(df, k=12):
    """Mined rubric per question: once on all data (for the report) and per fold
    using only that fold's training answers (leak-free prompt prefix for DeBERTa)."""
    df, _ = prepare(df)
    fold = make_folds(df)
    full, per_fold = {}, {}
    for q, d in df.groupby("q"):
        m = int(d.y.max())
        full[int(q)] = RubricMiner().fit(d, soft_target(d) / m, m).describe(k)
        for f in range(fold.max() + 1):
            tr = d[fold[d.index] != f]
            per_fold.setdefault(str(f), {})[str(int(q))] = RubricMiner().fit(tr, soft_target(tr) / m, m).describe(k)
    json.dump(full, open(os.path.join(OUT, "mined_rubric.json"), "w"), indent=1)
    json.dump(per_fold, open(os.path.join(OUT, "mined_rubric_folds.json"), "w"))
    for q, r in full.items():
        print(f"Q{q} credit: {', '.join(r['credit'][:8])}\n    misconception: {', '.join(r['misconception'][:6])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "rubric"])
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train.csv"))
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--target", choices=["soft", "hard"], default="hard",
                    help="hard = train on Score1 (default, best in CV); soft = mean of both raters")
    ap.add_argument("--metric", choices=["pooled", "perq"], default="pooled",
                    help="which QWK the cut points maximise: one QWK over all answers (pooled) "
                         "or the mean of per-question QWKs (perq). Ask the TA which one is graded.")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    data = load(a.data)
    if a.cmd == "cv":
        run_cv(data, a.ablation, a.jobs, a.target, a.metric)
    else:
        dump_rubrics(data)
