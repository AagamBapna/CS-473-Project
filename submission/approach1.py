"""Approach 1: KNN with TF-IDF (cosine similarity).

Text      : lowercase, tokenize, edit-distance spelling normalisation of rare tokens
            (Levenshtein <= 2 to a frequent training token), Porter stemming, unigrams,
            TF-IDF with sublinear tf, L2-normalised rows (dot product == cosine similarity).
KNN       : K=5 nearest graded training responses of the SAME question (cosine), similarity-
            weighted mean grade, offset -0.1, then snapped to the nearest allowed grade.
            A question with no graded examples falls back to neighbours from all questions.
            Settings were chosen by 5-fold CV (tuning/approach1/); see weights/approach1/config.json.

Artifacts : weights/approach1/model.npz   (idf, training TF-IDF rows, grades, question ids)
            weights/approach1/vocab.json  (vocabulary, frequent words for spelling)
            weights/approach1/config.json (settings + CV result)
            written by TfidfKNNModel.save() after fit(). Plain numpy/JSON, no pickled estimators.
            Inference re-implements the fitted TF-IDF transform; needs only numpy, scipy,
            scikit-learn (stop-word list), nltk (Porter stemmer).
"""
from pathlib import Path
from collections import Counter
import json
import re

import numpy as np
import scipy.sparse as sp
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# IMPORTANT INPUT / OUTPUT CONTRACT
# ---------------------------------
# During grading, the autograder calls:
#
#     model = load_model(weights_dir, config, assets)
#     predictions = model.predict(test_rows, questions)
#
# test_rows: [{"id": "123", "question": "1", "text": "student answer ..."}, ...]  (no labels)
# questions[row["question"]]["allowed_grades"] gives the permitted grades.
# predict() RETURNS a Python list [{"id": "123", "score": 2}, ...] - one finite int/float per
# input id, every id exactly once, no files written.

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

DEFAULT_PARAMS = {
    # representation
    "stopwords": False, "stem": True, "spell": True, "ngram_max": 1, "sublinear_tf": True, "min_df": 1,
    # neighbours
    "k": 5, "weighting": "similarity", "aggregate": "mean", "same_question": True, "offset": -0.1,
}


# ----------------------------------------------------------------------------- spelling
def levenshtein(a, b, cutoff):
    """Levenshtein distance, returning cutoff+1 as soon as it must exceed cutoff."""
    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            row_min = min(row_min, v)
        if row_min > cutoff:
            return cutoff + 1
        prev = cur
    return prev[-1]


def _deletes(word, max_del):
    out, frontier = {word}, {word}
    for _ in range(max_del):
        nxt = set()
        for s in frontier:
            for i in range(len(s)):
                nxt.add(s[:i] + s[i + 1:])
        frontier = nxt - out
        out |= nxt
    return out


class SpellIndex:
    """Map a rare token to the most frequent known token within Levenshtein distance <= 2.
    Candidates come from a symmetric-delete index (two strings within edit distance d share a
    variant obtained by deleting <= d characters from each), then exact distance is checked."""

    MIN_LEN, MAX_DIST, MIN_FREQ = 4, 2, 3

    def __init__(self, frequent):
        self.frequent = list(frequent)                      # most frequent first
        self.rank = {w: i for i, w in enumerate(self.frequent)}
        self.index = {}
        for i, w in enumerate(self.frequent):
            for d in _deletes(w, self.MAX_DIST):
                self.index.setdefault(d, []).append(i)
        self.cache = {}

    @classmethod
    def from_texts(cls, texts):
        counts = Counter(tok for t in texts for tok in TOKEN_RE.findall(t.lower()))
        return cls([w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    if c >= cls.MIN_FREQ and len(w) >= cls.MIN_LEN])

    def correct(self, tok):
        if tok in self.rank or len(tok) < self.MIN_LEN or tok.isdigit():
            return tok
        hit = self.cache.get(tok)
        if hit is not None:
            return hit
        cands = set()
        for d in _deletes(tok, self.MAX_DIST):
            cands.update(self.index.get(d, ()))
        best, best_d, best_rank = tok, self.MAX_DIST + 1, len(self.frequent)
        for i in cands:
            d = levenshtein(self.frequent[i], tok, self.MAX_DIST)
            if d < best_d or (d == best_d and i < best_rank):
                best, best_d, best_rank = self.frequent[i], d, i
        if best_d > self.MAX_DIST:
            best = tok
        self.cache[tok] = best
        return best


# ----------------------------------------------------------------------------- text -> TF-IDF
class Tokenizer:
    """lowercase -> tokenize -> [spell-normalise] -> [drop stop words] -> [stem]."""

    def __init__(self, params, frequent_words=None):
        self.p = params
        self.spell = SpellIndex(frequent_words) if (params["spell"] and frequent_words is not None) else None
        self.stemmer = PorterStemmer() if params["stem"] else None
        self._stem_cache = {}

    def fit(self, texts):
        if self.p["spell"]:
            self.spell = SpellIndex.from_texts(texts)
        return self

    def __call__(self, text):
        toks = TOKEN_RE.findall(text.lower())
        if self.spell is not None:
            toks = [self.spell.correct(t) for t in toks]
        if self.p["stopwords"]:
            toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
        if self.stemmer is not None:
            toks = [self._stem_cache.get(t) or self._stem_cache.setdefault(t, self.stemmer.stem(t)) for t in toks]
        return toks


def word_ngrams(toks, ngram_max):
    out = list(toks)
    for n in range(2, ngram_max + 1):
        out.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return out


def tfidf_transform(texts, tokenizer, vocabulary, idf, ngram_max, sublinear_tf):
    """Same output as a fitted scikit-learn TfidfVectorizer.transform (checked in training)."""
    indptr, indices, data = [0], [], []
    for t in texts:
        counts = Counter(vocabulary[g] for g in word_ngrams(tokenizer(t), ngram_max) if g in vocabulary)
        indices.extend(counts.keys())
        data.extend(counts.values())
        indptr.append(len(indices))
    X = sp.csr_matrix((np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64),
                       np.asarray(indptr, dtype=np.int64)), shape=(len(indptr) - 1, len(idf)))
    if sublinear_tf:
        X.data = 1.0 + np.log(X.data)
    X = X.multiply(np.asarray(idf, dtype=np.float64)[None, :]).tocsr()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sp.diags(1.0 / norms) @ X).tocsr()


# ----------------------------------------------------------------------------- KNN rule
def neighbor_table(sim, train_q, query_q, k, same_question):
    """Top-k neighbours per query row -> (idx, sims, valid). Questions are integer codes;
    a query whose question has no training rows keeps all columns."""
    sim = np.asarray(sim, dtype=np.float32)
    if same_question:
        sim = sim.copy()
        mask = train_q[None, :] != query_q[:, None]
        mask[~(~mask).any(axis=1)] = False
        sim[mask] = -np.inf
    k = min(k, sim.shape[1])
    part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    part_s = np.take_along_axis(sim, part, axis=1)
    order = np.argsort(-part_s, axis=1, kind="stable")
    idx = np.take_along_axis(part, order, axis=1)
    sims = np.take_along_axis(part_s, order, axis=1)
    valid = np.isfinite(sims)
    return idx, np.where(valid, sims, 0.0), valid


def permitted_matrix(allowed_rows):
    """(n, m) grades each row may receive, padded with +inf."""
    m = max(len(r) for r in allowed_rows)
    out = np.full((len(allowed_rows), m), np.inf)
    for i, r in enumerate(allowed_rows):
        out[i, :len(r)] = np.sort(np.asarray(r, dtype=float))
    return out


def snap(values, permitted):
    """Nearest permitted grade per row (clip + round in one step)."""
    return permitted[np.arange(len(values)), np.argmin(np.abs(permitted - values[:, None]), axis=1)]


def combine_neighbors(nb_labels, nb_sims, valid, p, permitted, grade_set):
    k = min(p["k"], nb_labels.shape[1])
    L, V = nb_labels[:, :k], valid[:, :k]
    if p["weighting"] == "similarity":
        W = np.clip(nb_sims[:, :k], 0.0, None) * V
        dead = W.sum(axis=1) <= 1e-12
        W[dead] = V[dead]
    else:
        W = V.astype(np.float32)
    wsum = W.sum(axis=1)
    wsum[wsum == 0] = 1.0
    wmean = (W * L).sum(axis=1) / wsum
    if p["aggregate"] == "mean":
        return snap(wmean + p["offset"], permitted)
    # weighted vote over grades, restricted to the permitted ones, ties toward the mean
    scores = np.stack([(W * (L == g)).sum(axis=1) for g in grade_set], axis=1)
    scores = scores - 1e-6 * np.abs(grade_set[None, :] - wmean[:, None])
    allowed = np.isclose(grade_set[None, None, :], permitted[:, :, None]).any(axis=1)
    pred = grade_set[np.argmax(np.where(allowed, scores, -np.inf), axis=1)]
    none = ~allowed.any(axis=1)
    if none.any():
        pred[none] = snap(wmean[none] + p["offset"], permitted[none])
    return pred


def _as_python_number(g):
    g = float(g)
    return int(g) if g.is_integer() else g


# ----------------------------------------------------------------------------- model
class TfidfKNNModel:
    CHUNK = 1000   # query rows per similarity block (memory ~ CHUNK x n_train x 4 bytes)

    def __init__(self, params, vocabulary, idf, frequent_words, X, grades, question_ids):
        self.p = {**DEFAULT_PARAMS, **params}
        self.vocabulary = vocabulary
        self.idf = np.asarray(idf, dtype=np.float64)
        self.tokenizer = Tokenizer(self.p, frequent_words)
        self.X = sp.csr_matrix(X, dtype=np.float32)
        self.XT = self.X.T.tocsc()
        self.y = np.asarray(grades, dtype=np.float64)
        self.q_names = np.asarray([str(q) for q in question_ids])
        self.q_codes, self.q = np.unique(self.q_names, return_inverse=True)
        self.q_lookup = {q: i for i, q in enumerate(self.q_codes)}
        self.grade_set = np.unique(self.y)
        self.grades_by_q = {q: np.unique(self.y[self.q_names == q]) for q in self.q_codes}

    def transform(self, texts):
        return tfidf_transform([t if isinstance(t, str) else "" for t in texts], self.tokenizer,
                               self.vocabulary, self.idf, self.p["ngram_max"], self.p["sublinear_tf"]).astype(np.float32)

    def allowed_for(self, qid, questions):
        meta = (questions or {}).get(qid)
        if isinstance(meta, dict) and meta.get("allowed_grades"):
            return np.asarray([float(g) for g in meta["allowed_grades"]])
        return self.grades_by_q.get(qid, self.grade_set)          # fallback: training grades

    def predict_grades(self, texts, question_ids, questions):
        qids = [str(q) for q in question_ids]
        qcodes = np.asarray([self.q_lookup.get(q, -1) for q in qids], dtype=np.int64)
        Xq = self.transform(texts)
        pred = np.empty(len(qids), dtype=np.float64)
        for s in range(0, len(qids), self.CHUNK):
            e = min(len(qids), s + self.CHUNK)
            sim = (Xq[s:e] @ self.XT).toarray()
            idx, sims, valid = neighbor_table(sim, self.q, qcodes[s:e], self.p["k"], self.p["same_question"])
            permitted = permitted_matrix([self.allowed_for(q, questions) for q in qids[s:e]])
            pred[s:e] = combine_neighbors(self.y[idx], sims, valid, self.p, permitted, self.grade_set)
        return pred

    def predict(self, test_rows, questions):
        if not test_rows:
            return []
        pred = self.predict_grades([r.get("text", "") for r in test_rows],
                                   [r["question"] for r in test_rows], questions)
        return [{"id": r["id"], "score": _as_python_number(s)} for r, s in zip(test_rows, pred)]

    # ---- artifacts (plain numpy + json)
    def save(self, out_dir, extra_config=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        X = self.X.tocsr()
        np.savez_compressed(out / "model.npz", idf=self.idf.astype(np.float32),
                            X_data=X.data.astype(np.float32), X_indices=X.indices.astype(np.int32),
                            X_indptr=X.indptr.astype(np.int64), X_shape=np.asarray(X.shape, dtype=np.int64),
                            grades=self.y.astype(np.float32), questions=np.asarray(self.q_names, dtype="U"))
        (out / "vocab.json").write_text(json.dumps({"vocabulary": self.vocabulary,
                                                    "frequent_words": self.tokenizer.spell.frequent if self.tokenizer.spell else None}))
        (out / "config.json").write_text(json.dumps({**self.p, **(extra_config or {})}, indent=2) + "\n")

    @classmethod
    def load(cls, wdir):
        wdir = Path(wdir)
        cfg = json.loads((wdir / "config.json").read_text())
        voc = json.loads((wdir / "vocab.json").read_text())
        d = np.load(wdir / "model.npz")
        X = sp.csr_matrix((d["X_data"], d["X_indices"], d["X_indptr"]), shape=tuple(int(v) for v in d["X_shape"]))
        return cls({k: cfg[k] for k in DEFAULT_PARAMS if k in cfg}, voc["vocabulary"], d["idf"],
                   voc["frequent_words"], X, d["grades"], d["questions"])


def _find_weights(weights_dir):
    here = Path(__file__).resolve().parent
    candidates = ([Path(weights_dir) / "approach1", Path(weights_dir)] if weights_dir else []) + [here / "weights" / "approach1"]
    for c in candidates:
        if (c / "model.npz").exists():
            return c
    raise FileNotFoundError(f"approach1 artifacts not found in {[str(c) for c in candidates]}")


def load_model(weights_dir, config, assets):
    # The grader calls this once, then calls predict() once on the evaluation set. Load only.
    return TfidfKNNModel.load(_find_weights(weights_dir))


# ----------------------------------------------------------------------------- tuning / experiments
def fit(train_rows, **params):
    """Build a model from graded rows, for CV tuning and Experiments 2-3 (not used by the grader).

    train_rows : list of {"question", "text", "score"} (score = Score1)
    params     : override any of DEFAULT_PARAMS, e.g. k=11, aggregate="vote", stopwords=True

        model = approach1.fit(train_rows, k=7)
        preds = model.predict(val_rows, questions)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    p = {**DEFAULT_PARAMS, **params}
    texts = [r["text"] if isinstance(r["text"], str) else "" for r in train_rows]
    tok = Tokenizer(p).fit(texts)
    vec = TfidfVectorizer(tokenizer=tok, preprocessor=lambda s: s, token_pattern=None, lowercase=False,
                          ngram_range=(1, p["ngram_max"]), sublinear_tf=p["sublinear_tf"], min_df=p["min_df"], norm="l2")
    X = vec.fit_transform(texts)
    model = TfidfKNNModel(p, {str(k): int(v) for k, v in vec.vocabulary_.items()}, vec.idf_,
                          tok.spell.frequent if tok.spell else None, X,
                          [r["score"] for r in train_rows], [r["question"] for r in train_rows])
    model._sklearn_vectorizer = vec     # lets train_approach1.py verify tfidf_transform
    return model
