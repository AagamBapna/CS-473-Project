"""Shared pieces for Experiments 1-3 (CS 47300 Part 1)."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "shim")); sys.path.insert(0, os.path.join(HERE, "team"))
sys.path.insert(0, "/home/claude/approach3"); sys.path.insert(0, "/home/claude/submission")
from common import qwk, load, make_folds  # noqa
DATA = "/home/claude/approach3/data/train_updated.csv"
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")


def load_data():
    df = load(DATA)                      # id, q, text, y (Score1), y2
    df["text"] = df.text.astype(str)
    return df


def allowed(df):
    out = {}
    for q, d in df.groupby("q"):
        step = 1.0 if (d.y == d.y.round()).all() else 0.5
        out[str(q)] = {"allowed_grades": [x if step == 0.5 else int(x) for x in np.arange(0, float(d.y.max()) + 1e-9, step)]}
    return out


def per_question_qwk(y, p, q):
    """Mean of per-question QWK (grades x2 -> integer categories); questions whose true grades
    are all identical are skipped (QWK undefined), exactly like the leaderboard."""
    vals = {}
    for k in np.unique(q):
        a = np.rint(np.asarray(y)[q == k] * 2).astype(int); b = np.rint(np.asarray(p)[q == k] * 2).astype(int)
        if len(set(a)) > 1:
            vals[k] = qwk(a, b, int(max(a.max(), b.max())) + 1)
    return float(np.mean(list(vals.values()))), vals


def split_TP(df, seed=0, frac=0.1):
    """Handout Exp 2: T (90%) / P (10%), with proportional coverage of every question and grade:
    within each question, answers are sorted by grade (random within grade) and every 10th goes to P."""
    rng = np.random.RandomState(seed)
    in_P = np.zeros(len(df), bool)
    for q in sorted(df.q.unique()):
        idx = np.where(df.q.values == q)[0]
        order = idx[np.lexsort((rng.rand(len(idx)), df.y.values[idx]))]
        start = rng.randint(int(1 / frac))
        in_P[order[start::int(1 / frac)]] = True
    return df[~in_P].copy(), df[in_P].copy()


def nested_subsets(T, seed, sizes=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Nested random training subsets of T: one random order per question (stratified by grade),
    subset k = first size_k fraction of every question. Returns {size: boolean mask over T}."""
    rng = np.random.RandomState(seed)
    rank = np.zeros(len(T))
    for q in T.q.unique():
        idx = np.where(T.q.values == q)[0]
        # deal answers of each grade evenly across the order so every subset covers every grade
        order = idx[np.lexsort((rng.rand(len(idx)), T.y.values[idx]))]
        pos = np.empty(len(idx)); pos[np.argsort(rng.rand(len(idx)) + np.arange(len(idx)) * 0)] = 0
        # interleave: position = (rank within grade) * (#grades) + grade offset, then normalise
        grades = T.y.values[order]
        within = np.zeros(len(order)); seen = {}
        for i, g in enumerate(grades):
            within[i] = seen.get(g, 0); seen[g] = within[i] + 1
        key = within + rng.rand(len(order)) * 0.5
        rank[order] = (np.argsort(np.argsort(key)) + 0.5) / len(order)
    return {s: rank <= s for s in sizes}


def rows(df):
    return [{"id": str(i), "question": str(q), "text": t, "score": float(y)} for i, q, t, y in zip(df.id, df.q, df.text, df.y)]


# ---------------- Approach 1 (teammate's TF-IDF KNN) ----------------
def a1_fit_predict(T, P, **params):
    import approach1
    model = approach1.fit(rows(T), **params)
    out = model.predict(rows(P), allowed(pd.concat([T, P])))
    return np.array([o["score"] for o in out], float)


# ---------------- Approach 2 (teammate's embedding KNN, precomputed MiniLM embeddings) ----------------
_EMB = None
def embeddings(df):
    global _EMB
    if _EMB is None:
        z = np.load(os.path.join(HERE, "team", "approach2", "train_embeddings.npz"))
        raw = pd.read_csv(DATA)
        _EMB = dict(zip(raw.Id, z["embeddings"].astype(np.float32)))
    return np.stack([_EMB[i] for i in df.id])


def a2_fit_predict(T, P, **params):
    import approach2
    cfg = {**approach2.DEFAULT_PARAMS, **params}
    model = approach2.EmbeddingKNNModel(embeddings(T), T.y.values, T.q.values, cfg)
    out = model.predict(rows(P), allowed(pd.concat([T, P])), embeddings=embeddings(P))
    return np.array([o["score"] for o in out], float)


# ---------------- Approach 3 (ours, CPU part: trained model, no LLM) ----------------
def a3_fit_predict(T, P, n_jobs=2):
    import tempfile, importlib.util, json
    from pathlib import Path
    import train_approach3 as TA
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp) / "weights" / "approach3"
        TA.export(T, w, n_jobs=n_jobs, check=False)
        spec = importlib.util.spec_from_file_location("a3entry", "/home/claude/submission/approach3.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        model = mod.load_model(str(Path(tmp) / "weights"), {"genai_enabled": False}, {})
        out = model.predict(rows(P), allowed(pd.concat([T, P])))
    got = {o["id"]: o["score"] for o in out}
    return np.array([got[str(i)] for i in P.id], float)
