"""Approach 3, small-question part: one cross-question model for questions with few graded
answers (the 0-5 questions with ~25 graded answers each). Shared by training and inference;
numpy/scipy only.

Each answer is described RELATIVE to its own question's graded answers (similarity to the
full-mark answers, grades of its nearest graded neighbours, coverage of the vocabulary the best
answers share, length relative to them), in three spaces: character n-grams, words, and LSA
(latent semantic analysis, which links answers that use related words). These features mean the same thing for every question,
so ONE ridge model is trained on all small questions together and applies to any of them.
Training rows get leave-one-out features (an answer is never compared with itself).
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

TOK = re.compile(r"[a-z]+|\d+")
STOP = set("""a an the of to in on at by for and or is are was were be been it its this that these those as with
from i we they he she you my our their them there then so do does did can will would should could has have had
which what when where who how not no yes""".split())


def stem(w):
    if len(w) <= 4 or w.isdigit():
        return w
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 3:
            return w[: len(w) - len(suf)] + rep
    return w


def clean(s):
    s = re.sub(r"<[^>]+>", " ", str(s))           # some answers contain <br>
    return " ".join(stem(t) for t in TOK.findall(s.lower()))


def content(s):
    return {t for t in s.split() if t not in STOP and len(t) > 1}


def sim_block(X_ref, X_q, t_ref, hi, lo, loo=False):
    """12 similarity features of query answers vs. one question's graded answers in one vector
    space (sparse TF-IDF or dense LSA)."""
    S = X_q @ X_ref.T
    S = S.toarray() if hasattr(S, "toarray") else np.array(S, dtype=float)
    if loo:
        np.fill_diagonal(S, -np.inf)
    valid = np.isfinite(S)
    Sv = np.where(valid, S, -1.0)
    order = np.argsort(-Sv, axis=1, kind="stable")
    f = []
    for k in (1, 3, 5):
        idx = order[:, :k]
        w = np.clip(np.take_along_axis(Sv, idx, 1), 0, None) + 1e-6
        f.append((w * t_ref[idx]).sum(1) / w.sum(1))
    for tau in (0.05, 0.15):
        E = np.where(valid, np.exp((Sv - Sv.max(1, keepdims=True)) / tau), 0)
        f.append((E * t_ref).sum(1) / E.sum(1))
    f.append(Sv.max(1))
    mx_hi = np.where(valid & hi, Sv, -1).max(1)
    mx_nhi = np.where(valid & ~hi, Sv, -1).max(1)
    mx_lo = np.where(valid & lo, Sv, -1).max(1)
    mean_hi = np.where(valid & hi, Sv, 0).sum(1) / np.maximum((valid & hi).sum(1), 1)
    f += [mx_hi, mx_nhi, mx_hi - mx_nhi, mx_lo, mx_hi - mx_lo, mean_hi]
    return np.c_[tuple(f)]


def hi_lo(t_ref):
    t_ref = np.asarray(t_ref, float)
    return t_ref >= t_ref.max() - 1e-9, t_ref <= np.quantile(t_ref, 0.25) + 1e-9


def lsa_features(Z_ref, t_ref, Z_q, loo=False):
    """Same 12 similarity features in the LSA ("meaning") space."""
    hi, lo = hi_lo(t_ref)
    return sim_block(Z_ref, Z_q, np.asarray(t_ref, float), hi, lo, loo)


def question_features(Xc_ref, Xw_ref, clean_ref, t_ref, Xc_q, Xw_q, clean_q, loo=False):
    """Features for query answers vs. one question's graded reference answers.
    t_ref: reference grades scaled to 0-1. loo=True: queries ARE the references (skip self)."""
    n = Xc_q.shape[0]
    t_ref = np.asarray(t_ref, float)
    top = t_ref.max()
    hi = t_ref >= top - 1e-9
    lo = t_ref <= np.quantile(t_ref, 0.25) + 1e-9
    prior = [t_ref.mean(), t_ref.std(), hi.mean(), np.median(t_ref), top]
    # vocabulary that the full-mark answers share (appears in >= 30% of them)
    hi_docs = [content(clean_ref[i]) for i in np.where(hi)[0]]
    cnt = Counter(w for d in hi_docs for w in d)
    key = {w for w, c in cnt.items() if c >= max(2, 0.3 * len(hi_docs))}
    any_hi = set().union(*hi_docs) if hi_docs else set()
    hi_len = np.median([len(clean_ref[i].split()) for i in np.where(hi)[0]]) if hi.any() else 1.0
    feats = [sim_block(X_ref, X_q, t_ref, hi, lo, loo) for X_ref, X_q in ((Xc_ref, Xc_q), (Xw_ref, Xw_q))]
    lex = []
    for s in clean_q:
        c = content(s)
        nw = len(s.split())
        lex.append([len(c & key) / max(len(key), 1), len(c & any_hi) / max(len(c), 1), float(len(c) == 0),
                    np.log1p(nw), np.log((nw + 1) / (hi_len + 1)), float(nw <= 2)])
    P = np.tile(prior, (n, 1))
    return np.hstack(feats + [np.array(lex).reshape(n, -1), P])


class SmallModel:
    """Loads weights/approach3/small/ and grades answers of any trained small question."""

    def __init__(self, folder):
        from a3_infer import Tfidf, word_ngrams, char_wb_ngrams
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        with np.load(folder / "arrays.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        self.vc = Tfidf(meta["char_vocab"], a["char_idf"], lambda s: char_wb_ngrams(s, 2, 5))
        self.vw = Tfidf(meta["word_vocab"], a["word_idf"], word_ngrams)
        self.a, self.scale, self.stretch = a, float(meta["scale"]), float(meta["stretch"])
        self.lc = Tfidf(meta["lsa_char_vocab"], a["lsa_char_idf"], lambda s: char_wb_ngrams(s, 2, 5))
        self.lw = Tfidf(meta["lsa_word_vocab"], a["lsa_word_idf"], word_ngrams)
        self.refs = {}
        for qid, r in meta["refs"].items():
            texts = r["clean"]
            self.refs[qid] = (self.vc.transform(texts), self.vw.transform(texts), texts,
                              np.asarray(r["t"], float), self.lsa(texts))

    def lsa(self, texts):
        from scipy import sparse
        X = sparse.hstack([self.lc.transform(texts), self.lw.transform(texts)]).tocsr()
        Z = np.asarray(X @ self.a["lsa_components"].T.astype(np.float64))
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def __contains__(self, qid):
        return qid in self.refs

    def predict_grades(self, qid, raw_texts):
        """Continuous grade on the original scale (e.g. 0-5), before snapping."""
        Xc_r, Xw_r, cl_r, t_r, Z_r = self.refs[qid]
        cl = [clean(s) for s in raw_texts]
        F = np.hstack([question_features(Xc_r, Xw_r, cl_r, t_r, self.vc.transform(cl), self.vw.transform(cl), cl),
                       lsa_features(Z_r, t_r, self.lsa(cl))])
        a = self.a
        p = ((F - a["mean"]) / a["scale_"]) @ a["coef"] + float(a["intercept"])
        prior = t_r.mean()
        p = prior + self.stretch * (p - prior)          # QWK rewards using the full grade range
        return np.clip(p, 0, 1) * self.scale
