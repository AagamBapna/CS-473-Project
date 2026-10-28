"""Shared utilities for Approach 3: data loading, fixed folds, QWK, threshold tuning.

Every teammate should use make_folds() with the same seed so all three approaches
are evaluated on identical folds (outputs/folds.csv).
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# Accept both the Kaggle-style headers and the handout's headers.
_ALIASES = {
    "id": ["Id", "id", "ID", "Submission Id", "submission_id"],
    "q": ["EssaySet", "Question Number", "Question", "question", "question_id"],
    "y": ["Score1", "Manual Score", "Score", "score", "grade"],
    "y2": ["Score2"],
    "text": ["EssayText", "Student Submission", "Text", "text", "answer"],
}


def load(path):
    """Load a CSV and rename columns to: id, q, text, y (Score1), y2 (Score2, optional).
    Unlabeled test files simply have no y column."""
    raw = pd.read_csv(path)
    df = pd.DataFrame()
    for key, names in _ALIASES.items():
        for n in names:
            if n in raw.columns:
                df[key] = raw[n]
                break
    for need in ("id", "q", "text"):
        if need not in df:
            raise ValueError(f"{path}: could not find a column for '{need}'. Columns: {list(raw.columns)}")
    df["text"] = df["text"].fillna("").astype(str)
    for c in ("y", "y2"):              # keep half-point grades (e.g. 2.5) as floats
        if c in df and df[c].notna().all() and (df[c] == df[c].round()).all():
            df[c] = df[c].astype(int)
    return df.reset_index(drop=True)


def make_folds(df, k=5, seed=0, path=None):
    """Stratified by question x grade, robust to questions with only ~25 answers: within each
    question, answers are sorted by grade (random order among equal grades) and dealt to the
    folds in turn, so every fold gets a proportional share of every question and grade.
    Saved so every approach reuses the same split."""
    path = path or os.path.join(OUT, "folds_v2.csv")
    if os.path.exists(path):
        f = pd.read_csv(path)
        m = dict(zip(f.id, f.fold))
        if len(m) == len(df) and all(i in m for i in df.id):
            return df.id.map(m).values.astype(int)
    rng = np.random.RandomState(seed)
    fold = np.zeros(len(df), int)
    start = 0
    for q in sorted(df.q.unique()):
        idx = np.where(df.q.values == q)[0]
        order = idx[np.lexsort((rng.rand(len(idx)), df.y.values[idx]))]
        fold[order] = (np.arange(len(order)) + start) % k
        start += len(order)          # rotate so small questions don't all start at fold 0
    pd.DataFrame({"id": df.id, "fold": fold}).to_csv(path, index=False)
    return fold


def qwk(y, p, n=None):
    """Quadratic weighted kappa for integer grades (fast numpy version)."""
    y = np.asarray(y, int)
    p = np.asarray(p, int)
    n = n or int(max(y.max(), p.max())) + 1
    if n < 2 or len(y) == 0:
        return 0.0
    O = np.bincount(y * n + p, minlength=n * n).reshape(n, n).astype(float)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    W = (i - j) ** 2 / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return 1.0 - (W * O).sum() / den if den > 0 else 0.0


def apply_thresholds(pred, th):
    return np.searchsorted(np.asarray(th), pred).astype(int)


def fit_thresholds(pred, y, m, passes=3, grid=60):
    """Coordinate search for the m cut points that maximise QWK.
    pred: continuous predictions on the 0..m grade scale. Returns sorted cut points."""
    pred = np.asarray(pred, float)
    y = np.asarray(y, int)
    th = np.arange(m) + 0.5
    best = qwk(y, apply_thresholds(pred, th), m + 1)
    lo_all, hi_all = min(pred.min(), 0) - 1e-3, max(pred.max(), m) + 1e-3
    for _ in range(passes):
        for i in range(m):
            lo = th[i - 1] if i > 0 else lo_all
            hi = th[i + 1] if i < m - 1 else hi_all
            for c in np.linspace(lo, hi, grid)[1:-1]:
                t2 = th.copy()
                t2[i] = c
                v = qwk(y, apply_thresholds(pred, t2), m + 1)
                if v > best + 1e-9:
                    best, th = v, t2
    return th


def fit_thresholds_pooled(preds, ys, ms, init=None, passes=2, grid=40):
    """Jointly tune every question's cut points to maximise QWK computed over ALL answers
    pooled together (the metric if the course scores one QWK over the whole test set).
    preds/ys/ms: lists with one entry per question (continuous preds on that question's
    0..m scale, true grades, max grade). Returns a list of cut-point arrays."""
    ths = [np.array(t, float) for t in init] if init else [np.arange(m) + 0.5 for m in ms]
    n = int(max(ms)) + 1
    y_all = np.concatenate(ys).astype(int)
    off = np.cumsum([0] + [len(p) for p in preds])
    cur = np.concatenate([apply_thresholds(p, t) for p, t in zip(preds, ths)])
    best = qwk(y_all, cur, n)
    for _ in range(passes):
        for q, (p, m) in enumerate(zip(preds, ms)):
            lo_all, hi_all = min(p.min(), 0) - 1e-3, max(p.max(), m) + 1e-3
            for i in range(m):
                lo = ths[q][i - 1] if i > 0 else lo_all
                hi = ths[q][i + 1] if i < m - 1 else hi_all
                for c in np.linspace(lo, hi, grid)[1:-1]:
                    t2 = ths[q].copy()
                    t2[i] = c
                    trial = cur.copy()
                    trial[off[q]:off[q + 1]] = apply_thresholds(p, t2)
                    v = qwk(y_all, trial, n)
                    if v > best + 1e-9:
                        best, ths[q], cur = v, t2, trial
    return ths


def soft_target(d):
    """Average of both raters when Score2 exists (less noisy), else Score1."""
    if "y2" in d and d["y2"].notna().all():
        return (d["y"].values + d["y2"].values) / 2.0
    return d["y"].values.astype(float)
