"""Feature components for the Approach 3 stacker.

Each component learns from ONE question's graded answers and turns answers into a few
numeric columns. All are generic: nothing is specific to a particular question.

Interface:  comp.fit(d, t, m) ; comp.transform(d) -> (n, k) array ; comp.names
    d: DataFrame slice with columns text, norm, row     t: soft target in [0,1]
    m: max grade for this question (read from the data)
"""
import re
from collections import Counter, defaultdict
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression

TOKEN = re.compile(r"[a-z]+|\d+")
# Used ONLY to skip n-grams made purely of function words; never removed from text
# (stop-word removal hurt in our baseline: "not", "more", "less" carry meaning).
FUNCTION = set("""a an the of to in on at by for and or is are was were be been it its this that these
those as with from i we they he she you my our their them there then so do does did""".split())


# ---------------------------------------------------------------- text normalisation
def light_stem(w):
    if len(w) <= 4 or w.isdigit():
        return w
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 3:
            return w[: len(w) - len(suf)] + rep
    return w


class SpellNormalizer:
    """Per-question spelling correction. The dictionary is the question's own frequent
    vocabulary (unlabeled text only), so it adapts to any new question and its jargon.
    Uses symmetric single-delete matching (SymSpell idea), which catches typos like
    'difussion' -> 'diffusion' or 'osmosious' -> 'osmosis' cheaply."""

    def __init__(self, min_freq=3, global_freq=None):
        self.min_freq = min_freq
        # word counts over ALL questions' answers: protects real words ('fast', 'rain')
        # that happen to be rare within one question from being "corrected"
        self.gf = global_freq or {}

    @staticmethod
    def _deletes(w):
        return {w[:i] + w[i + 1:] for i in range(len(w))}

    def fit(self, texts):
        cnt = Counter(t for s in texts for t in TOKEN.findall(s.lower()))
        self.freq = {w: c for w, c in cnt.items() if c >= self.min_freq and len(w) > 1}
        self.index = defaultdict(list)
        for w in self.freq:
            if len(w) >= 4:
                self.index[w].append(w)
                for d in self._deletes(w):
                    self.index[d].append(w)
        self.cache = {}
        return self

    @staticmethod
    def _lev(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def correct(self, w):
        """Replace w by a much more frequent word within edit distance 1 (2 for long words).
        A misspelling that is itself common ('difussion') is still fixed when the correct
        form is >=5x more frequent."""
        f = self.freq.get(w, 0)
        if len(w) < 4 or w.isdigit() or f >= 20:
            return w
        if w in self.cache:
            return self.cache[w]
        maxd = 1 if len(w) < 7 else 2
        keys = {w} | self._deletes(w)
        if maxd == 2:
            keys |= {x for d in self._deletes(w) for x in self._deletes(d)}
        cands = {c for k in keys for c in self.index.get(k, []) if c != w}
        cands = [c for c in sorted(cands) if self.freq[c] >= max(3, 5 * f) and self._lev(w, c) <= maxd
                 and (len(w) > 6 or c[0] == w[0])]  # short typos rarely change the first letter
        best = max(cands, key=lambda c: (self.freq[c], -self._lev(w, c))) if cands else w
        g = self.gf.get(w, 0)
        if best != w and g >= 3 and 5 * g > self.gf.get(best, 0):
            best = w
        self.cache[w] = best
        return best

    def normalize(self, s):
        toks = TOKEN.findall(s.lower())
        return " ".join(light_stem(self.correct(t)) for t in toks)

    def oov_rate(self, s):
        toks = TOKEN.findall(s.lower())
        return np.mean([t not in self.freq for t in toks]) if toks else 1.0


# ---------------------------------------------------------------- components
def global_counts(texts):
    return Counter(t for s in texts for t in TOKEN.findall(s.lower()))


class TfidfRidge:
    """Word (1-2 gram, spell-normalised) + character (2-5 gram, raw) TF-IDF -> ridge on soft target."""
    names = ["tfidf_ridge"]

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _X(self, d, fit=False):
        if fit:
            self.vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
            return sparse.hstack([self.vw.fit_transform(d.norm), self.vc.fit_transform(d.text.str.lower())]).tocsr()
        return sparse.hstack([self.vw.transform(d.norm), self.vc.transform(d.text.str.lower())]).tocsr()

    def fit(self, d, t, m):
        self.model = Ridge(alpha=self.alpha).fit(self._X(d, True), t)
        return self

    def transform(self, d):
        return self.model.predict(self._X(d))[:, None]


class TfidfLogit(TfidfRidge):
    """Same features, multinomial logistic regression on the integer grade.
    Gives a differently-shaped opinion (class probabilities) for the stacker."""
    names = ["logit_expected", "logit_p0", "logit_ptop"]

    def __init__(self, C=4.0):
        self.C = C

    def fit(self, d, t, m):
        self.m = m
        y = np.rint(d.y.values).astype(int)
        self.classes = np.unique(y)
        if len(self.classes) < 2:
            self.model = None
            self.const = y[0] / max(m, 1)
            return self
        self.model = LogisticRegression(C=self.C, max_iter=3000).fit(self._X(d, True), y)
        return self

    def transform(self, d):
        if self.model is None:
            return np.tile([self.const, 1.0, 0.0], (len(d), 1))
        P = self.model.predict_proba(self._X(d))
        exp = P @ self.classes / max(self.m, 1)
        p0 = P[:, 0] if self.classes[0] == 0 else np.zeros(len(d))
        ptop = P[:, -1] if self.classes[-1] == self.m else np.zeros(len(d))
        return np.c_[exp, p0, ptop]


class RubricMiner:
    """Automatically mined rubric (inspired by Ramachandran et al. 2015, who replaced the
    2012 Kaggle winner's hand-written regexes with mined patterns).

    For every 1-3 gram of spell-normalised text, test whether answers containing it score
    higher (or lower) than answers without it (Welch-style z score). Keep the strongest,
    non-redundant phrases as 'credit concepts' and 'misconception concepts'.
    Outputs a rubric score (ridge over concept presence) plus order-free coverage stats."""
    names = ["rubric_ridge", "rubric_pos_count", "rubric_pos_zsum", "rubric_neg_count",
             "rubric_top10_cov", "rubric_max_z"]

    def __init__(self, n_pos=40, n_neg=20, max_n=3, min_df_frac=0.01, alpha=3.0):
        self.n_pos, self.n_neg, self.max_n = n_pos, n_neg, max_n
        self.min_df_frac, self.alpha = min_df_frac, alpha

    def _grams(self, s):
        toks = s.split()
        out = set()
        for n in range(1, self.max_n + 1):
            for i in range(len(toks) - n + 1):
                g = toks[i:i + n]
                if all(w in FUNCTION for w in g):
                    continue
                out.add(" ".join(g))
        return out

    def fit(self, d, t, m):
        docs = [self._grams(s) for s in d.norm]
        N = len(docs)
        min_df = max(5, int(self.min_df_frac * N))
        inv = defaultdict(list)
        for i, g in enumerate(docs):
            for x in g:
                inv[x].append(i)
        t = np.asarray(t, float)
        tot, sd = t.sum(), t.std() + 1e-9
        stats = []
        for g, idx in inv.items():
            k = len(idx)
            if k < min_df or k > N - min_df:
                continue
            mi = t[idx].mean()
            mo = (tot - t[idx].sum()) / (N - k)
            z = (mi - mo) / (sd * np.sqrt(1 / k + 1 / (N - k)))
            stats.append((z, g, set(idx)))
        pos = self._dedupe(sorted([s for s in stats if s[0] > 0], key=lambda s: (-s[0], s[1])), self.n_pos)
        neg = self._dedupe(sorted([s for s in stats if s[0] < 0], key=lambda s: (s[0], s[1])), self.n_neg)
        self.pos = [(g, z) for z, g, _ in pos]
        self.neg = [(g, z) for z, g, _ in neg]
        self.concepts = [g for g, _ in self.pos] + [g for g, _ in self.neg]
        self.ridge = Ridge(alpha=self.alpha).fit(self._presence(docs), t) if self.concepts else None
        return self

    @staticmethod
    def _dedupe(ranked, k, jac=0.8):
        kept = []
        for z, g, s in ranked:
            if any(len(s & s2) / len(s | s2) > jac for _, _, s2 in kept):
                continue
            kept.append((z, g, s))
            if len(kept) == k:
                break
        return kept

    def _presence(self, docs):
        return np.array([[c in g for c in self.concepts] for g in docs], float).reshape(len(docs), len(self.concepts))

    def transform(self, d):
        docs = [self._grams(s) for s in d.norm]
        if not self.concepts:
            return np.zeros((len(d), len(self.names)))
        P = self._presence(docs)
        npos = len(self.pos)
        zpos = np.array([z for _, z in self.pos]) if npos else np.zeros(0)
        Pp, Pn = P[:, :npos], P[:, npos:]
        top = Pp[:, :10]
        return np.c_[
            self.ridge.predict(P),
            Pp.sum(1),
            Pp @ zpos if npos else np.zeros(len(d)),
            Pn.sum(1),
            top.mean(1) if top.shape[1] else np.zeros(len(d)),
            (Pp * zpos).max(1) if npos else np.zeros(len(d)),
        ]

    def describe(self, k=10):
        return {"credit": [g for g, _ in self.pos[:k]], "misconception": [g for g, _ in self.neg[:k]]}


class Exemplar:
    """Similarity of an answer to the closest graded answers at EACH grade level.
    'How much does this look like the best 3-point answers vs the best 0-point answers?'
    Works on character TF-IDF by default, or on any dense embedding matrix."""

    def __init__(self, emb=None, k=5, levels=4, tag="char"):
        self.emb, self.k, self.levels, self.tag = emb, k, levels, tag
        self.names = [f"ex_{tag}_g{g}" for g in range(levels)] + [f"ex_{tag}_knn", f"ex_{tag}_max"]

    def _vec(self, d, fit=False):
        if self.emb is not None:
            V = self.emb[d.row.values]
            return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        if fit:
            self.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
            return self.v.fit_transform(d.text.str.lower())
        return self.v.transform(d.text.str.lower())

    def fit(self, d, t, m):
        self.V, self.y, self.t, self.m = self._vec(d, True), d.y.values.astype(int), np.asarray(t), m
        return self

    def transform(self, d):
        S = self._vec(d) @ self.V.T
        S = S.toarray() if sparse.issparse(S) else np.asarray(S)
        out = np.zeros((len(d), self.levels + 2))
        for g in range(min(self.levels, self.m + 1)):
            cols = np.where(self.y == g)[0]
            if len(cols):
                kk = min(self.k, len(cols))
                out[:, g] = -np.sort(-S[:, cols], axis=1)[:, :kk].mean(1)
        kk = min(10, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        w = np.clip(np.take_along_axis(S, idx, 1), 0, None) + 1e-9
        out[:, -2] = (w * self.t[idx]).sum(1) / w.sum(1)
        out[:, -1] = S.max(1)
        return out


class EmbRidge:
    """Ridge on a dense sentence embedding (computed in Colab, loaded from .npy)."""

    def __init__(self, emb, tag, alpha=3.0):
        self.emb, self.alpha, self.names = emb, alpha, [f"emb_{tag}_ridge"]

    def fit(self, d, t, m):
        self.model = Ridge(alpha=self.alpha).fit(self.emb[d.row.values], t)
        return self

    def transform(self, d):
        return self.model.predict(self.emb[d.row.values])[:, None]


class Surface:
    """Label-free answer statistics."""
    names = ["log_words", "uniq_ratio", "log_chars", "oov_rate", "n_sent", "has_digit"]

    def __init__(self, speller):
        self.sp = speller

    def fit(self, d, t, m):
        return self

    def transform(self, d):
        rows = []
        for s in d.text:
            w = TOKEN.findall(s.lower())
            rows.append([np.log1p(len(w)), len(set(w)) / max(len(w), 1), np.log1p(len(s)),
                         self.sp.oov_rate(s), s.count(".") + s.count("!") + s.count("?"),
                         float(any(c.isdigit() for c in s))])
        return np.array(rows)
