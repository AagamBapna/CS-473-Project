"""Train Approach 3 on labeled data and export it for the Gradescope autograder.

  python train_approach3.py --data data/train.csv --out ../weights/approach3
  python train_approach3.py --cv          # 5-fold CV through the EXPORTED model (true submission path)

Exports plain arrays (.npz) + JSON, read by a3_infer.py with numpy/scipy only, so the
autograder's library versions do not matter. Target = Score1. Cut points are tuned per
question, matching the leaderboard metric (QWK per question, averaged).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, KFold

from common import load, make_folds, qwk, fit_thresholds
from features import SpellNormalizer, global_counts, TfidfRidge, TfidfLogit, RubricMiner, Exemplar, Surface
from stack import make_stacker, INNER, SEED

HERE = Path(__file__).resolve().parent
COMPONENTS = ["tfidf", "logit", "rubric", "exemplar", "surface"]   # column order = a3_infer.features()


def _make(name, speller):
    return {"tfidf": lambda: TfidfRidge(alpha=1.0), "logit": lambda: TfidfLogit(C=4.0),
            "rubric": lambda: RubricMiner(), "exemplar": lambda: Exemplar(),
            "surface": lambda: Surface(speller)}[name]


def train_question(d, G):
    """Fit everything for one question. d: that question's labeled rows."""
    d = d.copy().reset_index(drop=True)
    d["row"] = np.arange(len(d))
    sp = SpellNormalizer(global_freq=G).fit(d.text)
    d["norm"] = [sp.normalize(s) for s in d.text]
    m = int(d.y.max())
    t = d.y.values / m

    # level 1: full fits (exported) + inner out-of-fold features (to train the stacker honestly)
    strat = d.y.values if np.bincount(d.y.values).min() >= INNER else None
    splitter = (StratifiedKFold(INNER, shuffle=True, random_state=SEED) if strat is not None
                else KFold(INNER, shuffle=True, random_state=SEED))
    splits = list(splitter.split(d, strat))
    full, F = {}, []
    for name in COMPONENTS:
        make = _make(name, sp)
        full[name] = make().fit(d, t, m)
        k = full[name].transform(d.iloc[:1]).shape[1]
        Fo = np.zeros((len(d), k))
        for a, b in splits:
            Fo[b] = make().fit(d.iloc[a], t[a], m).transform(d.iloc[b])
        F.append(Fo)
    F = np.hstack(F)

    # level 2: ridge stacker + per-question cut points from its own out-of-fold predictions
    mk = make_stacker("ridge")
    oof = np.zeros(len(d))
    for a, b in KFold(INNER, shuffle=True, random_state=SEED + 1).split(F):
        oof[b] = mk().fit(F[a], t[a]).predict(F[b])
    st = mk().fit(F, t)
    th = fit_thresholds(oof * m, d.y.values, m)

    # ---- export ----
    tr, lg, rb, ex = full["tfidf"], full["logit"], full["rubric"], full["exemplar"]
    vocab = lambda v: [g for g, _ in sorted(v.vocabulary_.items(), key=lambda x: x[1])]
    # logistic/exemplar re-fit identical vectorizers on the same text; reuse the ridge ones
    assert vocab(lg.vw) == vocab(tr.vw) and vocab(lg.vc) == vocab(tr.vc) == vocab(ex.v)
    scaler, ridge = st.steps[0][1], st.steps[1][1]
    V = ex.V.tocsr().astype(np.float32)
    nw = len(tr.vw.idf_)
    if lg.model is None:
        raise ValueError("question has a single grade; cannot train")
    meta = {"m": m, "freq": sp.freq, "word_vocab": vocab(tr.vw), "char_vocab": vocab(tr.vc),
            "rubric_pos": [g for g, _ in rb.pos], "rubric_neg": [g for g, _ in rb.neg],
            "ex_k": ex.k, "ex_levels": ex.levels, "n_train": len(d),
            "grade_counts": np.bincount(d.y.values, minlength=m + 1).tolist(),
            "stacker_alpha": float(ridge.alpha_)}
    arr = {"word_idf": tr.vw.idf_, "char_idf": tr.vc.idf_,
           "ridge_coef": tr.model.coef_, "ridge_intercept": np.float64(tr.model.intercept_),
           "logit_coef": lg.model.coef_, "logit_intercept": lg.model.intercept_,
           "logit_classes": lg.classes.astype(float),
           "rubric_coef": rb.ridge.coef_ if rb.concepts else np.zeros(0),
           "rubric_intercept": np.float64(rb.ridge.intercept_ if rb.concepts else 0.0),
           "rubric_zpos": np.array([z for _, z in rb.pos], float),
           "ex_data": V.data.astype(np.float32), "ex_indices": V.indices.astype(np.int32),
           "ex_indptr": V.indptr.astype(np.int64), "ex_shape": np.array(V.shape),
           "ex_y": ex.y.astype(np.int64), "ex_t": np.asarray(ex.t, float),
           "st_mean": scaler.mean_, "st_scale": scaler.scale_, "st_coef": ridge.coef_,
           "st_intercept": np.float64(ridge.intercept_), "thresholds": np.asarray(th, float)}
    # reference features for a self-test after export (first 50 training answers, full-fit path)
    ref_rows = d.iloc[:50]
    ref = np.hstack([full[n].transform(ref_rows) for n in COMPONENTS])
    return meta, arr, {"texts": ref_rows.text.tolist(), "F": ref, "nw": nw}


MIN_BIG = 150     # questions with fewer graded answers (or half-point grades) use the cross-question model


def split_questions(df):
    big, small = [], []
    for q, d in df.groupby("q"):
        whole = bool((d.y == d.y.round()).all())
        (big if len(d) >= MIN_BIG and whole else small).append(q)
    return sorted(big), sorted(small)


def export(df, out, n_jobs=2, check=True):
    import shutil
    import small_q
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in list(out.glob("q_*")) + list(out.glob("llm_*.json")):
        f.unlink()
    shutil.rmtree(out / "small", ignore_errors=True)
    big, small = split_questions(df)
    G = global_counts(df.text)
    dfb = df[df.q.isin(big)].copy()
    dfb["y"] = dfb.y.astype(int)
    res = Parallel(n_jobs=n_jobs)(delayed(train_question)(dfb[dfb.q == q], G) for q in big)
    dist = []
    for q, (meta, arr, _) in zip(big, res):
        (out / f"q_{q}.json").write_text(json.dumps(meta))
        np.savez_compressed(out / f"q_{q}.npz", **arr)
        c = np.array(meta["grade_counts"], float)
        dist.append(np.interp(np.linspace(0, 1, 4), np.linspace(0, 1, len(c)), c / c.sum()))
    if small:
        ds = df[df.q.isin(small)]
        small_q.export(ds, out / "small", scale=float(ds.y.max()))
        # LLM stage: the instructor-graded answers of each small question, shown as examples
        ex = {str(q): [[str(t), float(g)] for t, g in zip(d.text, d.y)] for q, d in ds.groupby("q")}
        (out / "llm_examples.json").write_text(json.dumps(ex))
        sys.path.insert(0, str(HERE.parent))
        from a3_llm import DEFAULTS
        cfg_path = HERE / "llm_config.json"          # tuned settings live here (see llm_eval.py)
        cfg = {**DEFAULTS, **(json.loads(cfg_path.read_text()) if cfg_path.exists() else {})}
        (out / "llm_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "global_counts.json").write_text(json.dumps(dict(G)))
    (out / "index.json").write_text(json.dumps({
        "questions": [str(q) for q in big], "small_questions": [str(q) for q in small],
        "default_grade_dist": np.mean(dist, 0).tolist() if dist else [0.25, 0.25, 0.25, 0.25],
        "method": "Approach 3: per-question rubric-mined stacked grader for questions with many graded answers; "
                  "one cross-question similarity model for questions with few graded answers"}))
    if check:   # exported numpy path must reproduce the scikit-learn features exactly
        sys.path.insert(0, str(HERE.parent))
        import a3_infer
        model = a3_infer.Approach3Model(out)
        for q, (_, _, ref) in zip(big, res):
            got = model.models[str(q)].features(ref["texts"])
            err = np.abs(got - ref["F"]).max()
            assert err < 2e-3, f"question {q}: exported features differ from training by {err}"
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"exported {len(big)} per-question models + cross-question model for {len(small)} small questions "
          f"to {out} ({size / 1e6:.1f} MB)")


def allowed_grades(df):
    """Grade grid per question as the grader would list it: 0..max in steps of 1, or 0.5 if half
    points occur."""
    info = {}
    for q, d in df.groupby("q"):
        step = 1.0 if (d.y == d.y.round()).all() else 0.5
        top = float(d.y.max())
        info[str(q)] = {"allowed_grades": [x if step == 0.5 else int(x) for x in np.arange(0, top + 1e-9, step)]}
    return info


def per_question_qwk(y, p, q):
    """QWK per question (grades x2 so half points become integer categories). Questions whose
    true grades are all identical have no defined QWK and are skipped."""
    out = {}
    for k in np.unique(q):
        a = np.rint(y[q == k] * 2).astype(int)
        b = np.rint(p[q == k] * 2).astype(int)
        if len(set(a)) > 1:
            out[int(k)] = qwk(a, b, int(max(a.max(), b.max())) + 1)
    return out


def cv_exported(df, n_jobs=2):
    """Score the real submission path: export on 4 folds, run approach3.py on the 5th."""
    import importlib.util
    import tempfile
    sub = HERE.parent
    spec = importlib.util.spec_from_file_location("approach3_entry", sub / "approach3.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    fold = make_folds(df)
    qinfo = allowed_grades(df)
    big, small = split_questions(df)
    pred = np.zeros(len(df))
    for f in range(fold.max() + 1):
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            export(df[fold != f], Path(tmp) / "weights" / "approach3", n_jobs, check=(f == 0))
            model = entry.load_model(str(Path(tmp) / "weights"), {}, {})
            va = df[fold == f]
            rows = [{"id": str(i), "question": str(q), "text": t} for i, q, t in zip(va.id, va.q, va.text)]
            out = model.predict(rows, qinfo)
        got = {r["id"]: r["score"] for r in out}
        assert all(got[str(i)] in qinfo[str(q)]["allowed_grades"] for i, q in zip(va.id, va.q))
        pred[fold == f] = [got[str(i)] for i in va.id]
        print(f"fold {f} done [{time.time() - t0:.0f}s]", flush=True)
    y, q = df.y.values.astype(float), df.q.values
    pq = per_question_qwk(y, pred, q)
    qb = [pq[k] for k in big if k in pq]
    qs = [pq[k] for k in small if k in pq]
    fb = [np.mean([v for k, v in per_question_qwk(y[fold == f], pred[fold == f], q[fold == f]).items() if k in big])
          for f in range(fold.max() + 1)]
    print(f"SUBMISSION-PATH CV (QWK per question on out-of-fold predictions, then averaged):")
    print(f"  {len(qb)} large questions : {np.mean(qb):.4f}   (fold-by-fold {np.mean(fb):.4f} ± {np.std(fb):.4f})")
    if qs:
        print(f"  {len(qs)} small questions : {np.mean(qs):.4f}   ({len(small) - len(qs)} skipped: all answers same grade)")
    print(f"  all {len(pq)} questions, equal weight (leaderboard-style): {np.mean(list(pq.values())):.4f}")
    pd.DataFrame({"id": df.id, "q": q, "fold": fold, "y": y, "pred": pred}).to_csv(
        HERE / "outputs" / "oof_submission.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "train.csv"))
    ap.add_argument("--out", default=str(HERE.parent / "weights" / "approach3"))
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    data = load(a.data)
    if a.cv:
        cv_exported(data, a.jobs)
    else:
        t0 = time.time()
        export(data, a.out, a.jobs)
        print(f"done in {time.time() - t0:.0f}s")
