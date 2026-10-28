"""Training side of the small-question model (fits vocabularies, LSA, ridge; exports).
Feature code is shared with the submission: ../a3_small.py (repository root).

Tuning (5-fold CV x 3 fold seeds, mean per-question QWK on the 80 scorable small questions):
  char + word similarity features, ridge                     0.470
  + stretch 1.5 around the question's mean grade             0.500
  + LSA-100 similarity features (char+word TF-IDF -> SVD)    0.533   <- this model
  tried and rejected: per-question isotonic/kernel calibration (0.41-0.44), pairwise
  "grade by comparison" model (0.500), ridge+boosting blend (0.497), LSA-50 (0.526),
  LSA-200 (0.523), LSA fitted with the large questions' texts too (0.504), word-only LSA
  (0.512), self-trained PPMI word vectors (0.497), stretch 1.3 / 1.7 / 2.0 (0.527 / 0.534 / 0.524).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SUB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUB))
from a3_small import clean, question_features, lsa_features  # noqa: E402

STRETCH = 1.5      # chosen by 5-fold CV over 3 fold seeds (1.5 and 1.7 tie; 1.5 is the safer choice)
LSA_DIMS = 100


def make_vectorizers(texts_clean):
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1).fit(texts_clean)
    vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(texts_clean)
    return vc, vw


class LSA:
    """Latent semantic space: char+word TF-IDF -> truncated SVD -> unit-length vectors."""

    def __init__(self, texts_clean, dims=LSA_DIMS):
        self.lc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2).fit(texts_clean)
        self.lw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(texts_clean)
        self.svd = TruncatedSVD(dims, random_state=0).fit(self._X(texts_clean))
        # store float32 components and use exactly those, so training == inference
        self.components = self.svd.components_.astype(np.float32)

    def _X(self, texts):
        return sparse.hstack([self.lc.transform(texts), self.lw.transform(texts)]).tocsr()

    def embed(self, texts):
        Z = np.asarray(self._X(texts) @ self.components.T.astype(np.float64))
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def build(df_ref, df_query, vc, vw, lsa, loo):
    """Features for df_query rows (columns q, clean, t) against same-question rows of df_ref."""
    res = None
    for q, dq in df_query.groupby("q", sort=False):
        dr = df_ref[df_ref.q == q]
        Xc_r, Xw_r = vc.transform(dr.clean), vw.transform(dr.clean)
        Xc_q, Xw_q = (Xc_r, Xw_r) if loo else (vc.transform(dq.clean), vw.transform(dq.clean))
        Z_r = lsa.embed(dr.clean)
        Z_q = Z_r if loo else lsa.embed(dq.clean)
        F = np.hstack([question_features(Xc_r, Xw_r, dr.clean.tolist(), dr.t.values, Xc_q, Xw_q,
                                         dq.clean.tolist(), loo=loo),
                       lsa_features(Z_r, dr.t.values, Z_q, loo=loo)])
        if res is None:
            res = np.zeros((len(df_query), F.shape[1]))
        res[df_query.index.get_indexer(dq.index)] = F
    return res


def fit(df, scale):
    """df: rows of small questions with columns q, text, y (original grade scale)."""
    df = df.copy()
    df["clean"] = [clean(s) for s in df.text]
    df["t"] = df.y.values / scale
    vc, vw = make_vectorizers(df.clean)
    lsa = LSA(df.clean)
    F = build(df, df, vc, vw, lsa, loo=True)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(F, df.t.values)
    return df, vc, vw, lsa, model


def export(df, out, scale):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df, vc, vw, lsa, model = fit(df, scale)
    vocab = lambda v: [g for g, _ in sorted(v.vocabulary_.items(), key=lambda x: x[1])]
    sc, rg = model.steps[0][1], model.steps[1][1]
    meta = {"scale": scale, "stretch": STRETCH, "char_vocab": vocab(vc), "word_vocab": vocab(vw),
            "lsa_char_vocab": vocab(lsa.lc), "lsa_word_vocab": vocab(lsa.lw), "lsa_dims": LSA_DIMS,
            "alpha": float(rg.alpha_),
            "refs": {str(q): {"clean": d.clean.tolist(), "t": d.t.round(6).tolist()} for q, d in df.groupby("q")}}
    (out / "meta.json").write_text(json.dumps(meta))
    np.savez_compressed(out / "arrays.npz", char_idf=vc.idf_, word_idf=vw.idf_, mean=sc.mean_, scale_=sc.scale_,
                        coef=rg.coef_, intercept=np.float64(rg.intercept_),
                        lsa_char_idf=lsa.lc.idf_, lsa_word_idf=lsa.lw.idf_, lsa_components=lsa.components)
    # self-test: exported numpy path == sklearn path on a few questions (non-LOO, as at test time)
    from a3_small import SmallModel
    sm = SmallModel(out)
    for q in list(df.q.unique())[:5]:
        dq = df[df.q == q].iloc[:6]
        p = model.predict(build(df, dq, vc, vw, lsa, loo=False))
        prior = df[df.q == q].t.mean()
        ref = np.clip(prior + STRETCH * (p - prior), 0, 1) * scale
        got = sm.predict_grades(str(q), dq.text.tolist())
        assert np.abs(got - ref).max() < 1e-6, (q, got, ref)
    return out
