"""Approach 3 inference: Rubric-Mined Stacked Grader.

Pure numpy/scipy re-implementation of the trained pipeline, so the saved artifacts
(plain .npz arrays + .json) load on any library version. Nothing is fitted here.

Per question:
  spelling normalisation -> 5 level-1 graders -> ridge stacker -> tuned cut points
Level-1 graders (22 features): TF-IDF ridge, TF-IDF logistic regression, mined rubric,
exemplar similarity per grade level, surface statistics.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

TOKEN = re.compile(r"[a-z]+|\d+")
SK_WORD = re.compile(r"(?u)\b\w\w+\b")          # scikit-learn's default token_pattern
WHITE = re.compile(r"\s\s+")
FUNCTION = set("""a an the of to in on at by for and or is are was were be been it its this that these
those as with from i we they he she you my our their them there then so do does did""".split())


# ------------------------------------------------------------------ text normalisation
def light_stem(w):
    if len(w) <= 4 or w.isdigit():
        return w
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 3:
            return w[: len(w) - len(suf)] + rep
    return w


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _deletes(w):
    return {w[:i] + w[i + 1:] for i in range(len(w))}


class Speller:
    """Same rules as training (features.SpellNormalizer); dictionary = saved word counts."""

    def __init__(self, freq, gf):
        self.freq, self.gf, self.cache = freq, gf, {}
        self.index = defaultdict(list)
        for w in sorted(freq):
            if len(w) >= 4:
                self.index[w].append(w)
                for d in _deletes(w):
                    self.index[d].append(w)

    def correct(self, w):
        f = self.freq.get(w, 0)
        if len(w) < 4 or w.isdigit() or f >= 20:
            return w
        if w in self.cache:
            return self.cache[w]
        maxd = 1 if len(w) < 7 else 2
        keys = {w} | _deletes(w)
        if maxd == 2:
            keys |= {x for d in _deletes(w) for x in _deletes(d)}
        cands = {c for k in keys for c in self.index.get(k, []) if c != w}
        cands = [c for c in sorted(cands) if self.freq[c] >= max(3, 5 * f) and _lev(w, c) <= maxd
                 and (len(w) > 6 or c[0] == w[0])]
        best = max(cands, key=lambda c: (self.freq[c], -_lev(w, c))) if cands else w
        g = self.gf.get(w, 0)
        if best != w and g >= 3 and 5 * g > self.gf.get(best, 0):
            best = w
        self.cache[w] = best
        return best

    def normalize(self, s):
        return " ".join(light_stem(self.correct(t)) for t in TOKEN.findall(s.lower()))

    def oov_rate(self, s):
        toks = TOKEN.findall(s.lower())
        return float(np.mean([t not in self.freq for t in toks])) if toks else 1.0


# ------------------------------------------------------------------ TF-IDF (scikit-learn compatible)
def word_ngrams(s, n_max=2):
    toks = SK_WORD.findall(s.lower())
    out = list(toks)
    for n in range(2, n_max + 1):
        out += [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return out


def char_wb_ngrams(s, n_min=2, n_max=5):
    out = []
    for w in WHITE.sub(" ", s.lower()).split():
        w = " " + w + " "
        L = len(w)
        for n in range(n_min, n_max + 1):
            off = 0
            out.append(w[off:off + n])
            while off + n < L:
                off += 1
                out.append(w[off:off + n])
            if off == 0:
                break
    return out


class Tfidf:
    """Sublinear TF x IDF, L2-normalised rows, fixed vocabulary (as TfidfVectorizer.transform)."""

    def __init__(self, vocab, idf, analyzer):
        self.vocab = {g: i for i, g in enumerate(vocab)}
        self.idf = np.asarray(idf, float)
        self.analyzer = analyzer

    def transform(self, texts):
        rows, cols, vals = [], [], []
        for r, s in enumerate(texts):
            cnt = defaultdict(int)
            for g in self.analyzer(s):
                j = self.vocab.get(g)
                if j is not None:
                    cnt[j] += 1
            for j, c in cnt.items():
                rows.append(r)
                cols.append(j)
                vals.append((1.0 + np.log(c)) * self.idf[j])
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(self.idf)))
        norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        norm[norm == 0] = 1.0
        return sparse.diags(1.0 / norm) @ X


# ------------------------------------------------------------------ one question's model
class QuestionModel:
    def __init__(self, meta, arr, gf):
        self.m = int(meta["m"])
        self.speller = Speller(meta["freq"], gf)
        self.word = Tfidf(meta["word_vocab"], arr["word_idf"], word_ngrams)
        self.char = Tfidf(meta["char_vocab"], arr["char_idf"], char_wb_ngrams)
        self.a = arr
        self.meta = meta
        self.rub_pos = meta["rubric_pos"]
        self.rub_concepts = meta["rubric_pos"] + meta["rubric_neg"]
        self.V = sparse.csr_matrix((arr["ex_data"].astype(np.float64), arr["ex_indices"], arr["ex_indptr"]),
                                   shape=tuple(arr["ex_shape"]))

    # level-1 graders -------------------------------------------------
    def _grams(self, s, max_n=3):
        toks = s.split()
        out = set()
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                g = toks[i:i + n]
                if not all(w in FUNCTION for w in g):
                    out.add(" ".join(g))
        return out

    def features(self, texts):
        a, m = self.a, self.m
        norm = [self.speller.normalize(s) for s in texts]
        Xw = self.word.transform(norm)
        Xc = self.char.transform([s.lower() for s in texts])
        X = sparse.hstack([Xw, Xc]).tocsr()
        n = len(texts)

        f_ridge = X @ a["ridge_coef"] + float(a["ridge_intercept"])

        Z = X @ a["logit_coef"].T + a["logit_intercept"]
        classes = a["logit_classes"]
        if Z.shape[1] == 1:                               # binary logistic regression
            p1 = 1.0 / (1.0 + np.exp(-Z[:, 0]))
            P = np.c_[1 - p1, p1]
        else:
            Z = Z - Z.max(1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(1, keepdims=True)
        f_logit = np.c_[P @ classes / max(m, 1),
                        P[:, 0] if classes[0] == 0 else np.zeros(n),
                        P[:, -1] if classes[-1] == m else np.zeros(n)]

        if self.rub_concepts:
            docs = [self._grams(s) for s in norm]
            C = np.array([[c in g for c in self.rub_concepts] for g in docs], float).reshape(n, -1)
            npos = len(self.rub_pos)
            zpos = a["rubric_zpos"]
            Pp, Pn = C[:, :npos], C[:, npos:]
            top = Pp[:, :10]
            f_rub = np.c_[C @ a["rubric_coef"] + float(a["rubric_intercept"]), Pp.sum(1),
                          Pp @ zpos if npos else np.zeros(n), Pn.sum(1),
                          top.mean(1) if top.shape[1] else np.zeros(n),
                          (Pp * zpos).max(1) if npos else np.zeros(n)]
        else:
            f_rub = np.zeros((n, 6))

        S = (Xc @ self.V.T).toarray()
        ys, ts = a["ex_y"], a["ex_t"]
        k, levels = int(self.meta["ex_k"]), int(self.meta["ex_levels"])
        f_ex = np.zeros((n, levels + 2))
        for g in range(min(levels, m + 1)):
            cols = np.where(ys == g)[0]
            if len(cols):
                kk = min(k, len(cols))
                f_ex[:, g] = -np.sort(-S[:, cols], axis=1)[:, :kk].mean(1)
        kk = min(10, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        w = np.clip(np.take_along_axis(S, idx, 1), 0, None) + 1e-9
        f_ex[:, -2] = (w * ts[idx]).sum(1) / w.sum(1)
        f_ex[:, -1] = S.max(1)

        f_surf = []
        for s in texts:
            ww = TOKEN.findall(s.lower())
            f_surf.append([np.log1p(len(ww)), len(set(ww)) / max(len(ww), 1), np.log1p(len(s)),
                           self.speller.oov_rate(s), s.count(".") + s.count("!") + s.count("?"),
                           float(any(c.isdigit() for c in s))])
        return np.hstack([f_ridge[:, None], f_logit, f_rub, f_ex, np.array(f_surf).reshape(n, 6)])

    def predict(self, texts):
        """Continuous score on the 0..m scale and integer grade."""
        F = self.features(texts)
        a = self.a
        z = ((F - a["st_mean"]) / a["st_scale"]) @ a["st_coef"] + float(a["st_intercept"])
        cont = z * self.m
        grade = np.clip(np.searchsorted(a["thresholds"], cont), 0, self.m)
        return cont, grade.astype(int)


# ------------------------------------------------------------------ whole model
def _num(v):
    """Ordinary Python int when the grade is whole, else float (e.g. 2.5)."""
    v = float(v)
    return int(round(v)) if abs(v - round(v)) < 1e-9 else v


def _snap(g, allowed):
    """Map a predicted grade to the nearest permitted grade value (ties -> the higher one)."""
    allowed = sorted(float(x) for x in allowed)
    return _num(min(allowed, key=lambda v: (abs(v - g), -v)))


class Approach3Model:
    def __init__(self, folder):
        folder = Path(folder)
        index = json.loads((folder / "index.json").read_text())
        gf = json.loads((folder / "global_counts.json").read_text())
        self.default_dist = index["default_grade_dist"]
        self.small = None
        if (folder / "small" / "meta.json").exists():
            from a3_small import SmallModel
            self.small = SmallModel(folder / "small")
        self.models = {}
        for qid in index["questions"]:
            meta = json.loads((folder / f"q_{qid}.json").read_text())
            with np.load(folder / f"q_{qid}.npz", allow_pickle=False) as z:
                arr = {k: z[k] for k in z.files}
            self.models[str(qid)] = QuestionModel(meta, arr, gf)

    @staticmethod
    def _allowed(questions, qid, fallback_m):
        info = (questions or {}).get(qid) or (questions or {}).get(str(qid)) or {}
        allowed = info.get("allowed_grades") if isinstance(info, dict) else None
        if not allowed:
            allowed = list(range(fallback_m + 1))
        return [float(x) for x in allowed]

    def _unseen(self, texts, allowed):
        """Question we were never trained on: rank answers by length/vocabulary richness
        (longer, richer answers tend to score higher) and assign grades by quantile, using the
        average grade distribution of the training questions. Weak but valid."""
        allowed = sorted(allowed)
        score = np.array([np.log1p(len(TOKEN.findall(s.lower()))) + 0.5 * np.log1p(len(set(TOKEN.findall(s.lower()))))
                          for s in texts])
        dist = np.interp(np.linspace(0, 1, len(allowed)), np.linspace(0, 1, len(self.default_dist)),
                         self.default_dist)
        cuts = np.cumsum(dist / dist.sum())[:-1]
        ranks = (np.argsort(np.argsort(score, kind="stable"), kind="stable") + 0.5) / max(len(texts), 1)
        return [_num(allowed[int(np.searchsorted(cuts, r))]) for r in ranks]

    def predict(self, test_rows, questions):
        groups = defaultdict(list)
        for i, r in enumerate(test_rows):
            groups[str(r["question"])].append(i)
        scores = [None] * len(test_rows)
        for qid, idx in groups.items():
            texts = ["" if test_rows[i].get("text") is None else str(test_rows[i]["text"]) for i in idx]
            qm = self.models.get(qid)
            if qm is None and self.small is not None and qid in self.small:
                # few graded answers -> cross-question model (continuous grade, then snap)
                allowed = self._allowed(questions, qid, int(self.small.scale))
                try:
                    preds = [_snap(x, allowed) for x in self.small.predict_grades(qid, texts)]
                except Exception:
                    preds = self._unseen(texts, allowed)
            elif qm is None:
                allowed = self._allowed(questions, qid, 2)
                preds = self._unseen(texts, allowed)
            else:
                allowed = self._allowed(questions, qid, qm.m)
                try:
                    _, g = qm.predict(texts)
                    g = [int(x) for x in g]
                except Exception:          # never fail the whole approach on one question
                    g = self._unseen(texts, allowed)
                preds = [_snap(x, allowed) for x in g]
            for i, p in zip(idx, preds):
                scores[i] = _num(p)
        return [{"id": r["id"], "score": s} for r, s in zip(test_rows, scores)]
