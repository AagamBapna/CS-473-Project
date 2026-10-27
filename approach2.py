"""Approach 2: KNN with Embeddings (Sentence-BERT).

Embedding : sentence-transformers/all-MiniLM-L6-v2 (staff-supplied "minilm" asset, pinned
            revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41), mean pooling, L2-normalised, CPU.
KNN       : cosine similarity against the stored graded training responses of the SAME question,
            after subtracting that question's mean training embedding.
            Large questions (>= 100 stored responses): K=11, similarity-weighted vote.
            Small questions (< 100 stored responses):  K=3,  similarity-weighted mean,
            mapped to the nearest allowed grade (exact midpoints round up).
Default K / combination settings are tunable via fit(..., **params) for the team CV run.

Artifacts : weights/approach2/train_embeddings.npz, weights/approach2/config.json
            (precomputed offline from training_dataset.csv with the same encoder).
"""
from pathlib import Path
import json

import numpy as np

# IMPORTANT INPUT / OUTPUT CONTRACT
# ---------------------------------
# During grading, the autograder calls:
#
#     model = load_model(weights_dir, config, assets)
#     predictions = model.predict(test_rows, questions)
#
# test_rows is a list like:
#     [
#         {"id": "123", "question": "1", "text": "student answer ..."},
#         {"id": "456", "question": "11", "text": "another answer ..."},
#     ]
#
# There is NO Score1 or Score2 in test_rows. Evaluation labels are hidden.
#
# questions is keyed by question id. At minimum you should use:
#     questions[row["question"]]["allowed_grades"]
# for rubric labels and range endpoints. Any finite int/float in that range is valid.
#
# predict() must RETURN a Python list -- do NOT write prediction files:
#     [
#         {"id": "123", "score": 2},
#         {"id": "456", "score": 3.5},
#     ]
#
# Preserve every input id exactly once and return a finite Python int or float.


# ----------------------------------------------------------------------------- encoder
class MiniLMEncoder:
    """all-MiniLM-L6-v2 sentence embeddings (mean pooling + L2 normalisation).

    Uses sentence-transformers when available (as on the grader); otherwise runs the ONNX export
    in the same model folder (onnx/model.onnx + tokenizer.json) with onnxruntime. Both give the
    same embeddings."""

    def __init__(self, model_path, threads=4, max_len=256):
        self.model_path = str(model_path)
        self.threads = int(threads)
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(self.threads)
            self.st = SentenceTransformer(self.model_path, device="cpu",
                                          local_files_only=True, trust_remote_code=False)
            self.st.max_seq_length = max_len
        except ImportError:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            self.st = None
            self.tok = Tokenizer.from_file(str(Path(self.model_path) / "tokenizer.json"))
            self.tok.enable_truncation(max_len)
            self.tok.enable_padding()
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            self.sess = ort.InferenceSession(str(Path(self.model_path) / "onnx" / "model.onnx"), so,
                                             providers=["CPUExecutionProvider"])
            self.input_names = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts, batch_size=64):
        texts = [t if isinstance(t, str) else "" for t in texts]
        if self.st is not None:
            import torch
            with torch.inference_mode():
                X = self.st.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False)
            return X.astype(np.float32)
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in enc], np.int64)
            mask = np.array([e.attention_mask for e in enc], np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            e = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None))
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)


def minilm_path(assets, weights_dir=None):
    """Location of the staff-supplied MiniLM encoder (assets['models']['minilm']['path'])."""
    try:
        return assets["models"]["minilm"]["path"]
    except (KeyError, TypeError):
        raise RuntimeError("Approach 2 needs the 'minilm' encoder asset "
                           "(local testing: python local_test.py ... --assets local_assets/assets.json)")


# ----------------------------------------------------------------------------- KNN helpers
def _l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def nearest_allowed(x, allowed):
    """Nearest allowed grade; an exact midpoint between two grades rounds up."""
    allowed = np.asarray(allowed, dtype=float)
    d = np.abs(allowed[None, :] - np.asarray(x, dtype=float)[:, None])
    best = d.min(1, keepdims=True)
    # among (near-)ties pick the highest grade
    cand = np.where(np.isclose(d, best), allowed[None, :], -np.inf)
    return cand.max(1)


def knn_grades(Xtr, ytr, Xte, K, rule, allowed, center=True):
    if center:
        mu = Xtr.mean(0, keepdims=True)
        Xtr, Xte = Xtr - mu, Xte - mu
    S = _l2norm(Xte) @ _l2norm(Xtr).T
    K = max(1, min(int(K), S.shape[1]))
    idx = np.argsort(-S, axis=1, kind="stable")[:, :K]
    sims = np.clip(np.take_along_axis(S, idx, 1), 1e-6, None)   # similarity weights
    grades = ytr[idx]
    if rule == "vote":
        allowed = np.asarray(allowed, dtype=float)
        votes = np.stack([(sims * np.isclose(grades, a)).sum(1) for a in allowed], 1)
        nearest = np.argmin(np.abs(allowed[None, :] - grades[:, :1]), 1)
        votes[np.arange(len(votes)), nearest] += 1e-9               # tie -> nearest neighbour's grade
        return allowed[votes.argmax(1)]
    return nearest_allowed((sims * grades).sum(1) / sims.sum(1), allowed)


def _as_python_number(g):
    g = float(g)
    return int(g) if g.is_integer() else g


# ----------------------------------------------------------------------------- model
DEFAULT_PARAMS = {"center": True, "large_min": 100,
                  "k_large": 11, "combine_large": "vote",
                  "k_small": 3, "combine_small": "wmean", "max_seq_length": 256}


class EmbeddingKNNModel:
    def __init__(self, X, grades, question_ids, params, encoder=None):
        X = np.asarray(X, np.float32); y = np.asarray(grades, float)
        q = np.asarray([str(v) for v in question_ids])
        self.train = {qid: (X[q == qid], y[q == qid]) for qid in np.unique(q)}
        self.cfg = {**DEFAULT_PARAMS, **params}
        self.encoder = encoder

    def predict(self, test_rows, questions, embeddings=None):
        """embeddings (optional, not used by the grader): precomputed MiniLM embeddings of
        test_rows, in the same order, to skip re-encoding during tuning."""
        if not test_rows:
            return []
        qids = np.array([str(r["question"]) for r in test_rows])
        X = (np.asarray(embeddings, np.float32) if embeddings is not None
             else self.encoder.encode([r.get("text", "") for r in test_rows]))
        scores = np.zeros(len(test_rows))
        for qid in np.unique(qids):
            m = qids == qid
            allowed = sorted(float(a) for a in questions[qid]["allowed_grades"])
            if qid not in self.train:                       # no graded examples: middle grade
                scores[m] = allowed[len(allowed) // 2]
                continue
            Xtr, ytr = self.train[qid]
            if len(ytr) >= self.cfg["large_min"]:
                K, rule = self.cfg["k_large"], self.cfg["combine_large"]
            else:
                K, rule = self.cfg["k_small"], self.cfg["combine_small"]
            scores[m] = knn_grades(Xtr, ytr, X[m], K, rule, allowed, self.cfg.get("center", True))
        return [{"id": r["id"], "score": _as_python_number(s)} for r, s in zip(test_rows, scores)]


def load_model(weights_dir, config, assets):
    # The grader calls this once, then calls predict() once on the evaluation set.
    wdir = Path(weights_dir) / "approach2"
    cfg = json.loads((wdir / "config.json").read_text())
    data = np.load(wdir / "train_embeddings.npz")
    encoder = MiniLMEncoder(minilm_path(assets), threads=(config or {}).get("threads", 4),
                            max_len=cfg.get("max_seq_length", 256))
    return EmbeddingKNNModel(data["embeddings"], data["grades"], data["questions"], cfg, encoder)


# ----------------------------------------------------------------------------- tuning / experiments
def fit(train_rows, assets, embeddings=None, **params):
    """Build a model from graded rows, for CV tuning and Experiments 2-3 (not used by the grader).

    train_rows : list of {"question", "text", "score"} (score = Score1)
    assets     : {"models": {"minilm": {"path": ...}}}  (local_assets/assets.json)
    embeddings : optional precomputed embeddings of train_rows (see encode()) - saves re-encoding
    params     : override any of DEFAULT_PARAMS, e.g. k_large=21, k_small=5, combine_small="vote"

        model = approach2.fit(train_rows, assets, k_small=5)
        preds = model.predict(val_rows, questions)
    """
    cfg = {**DEFAULT_PARAMS, **params}
    encoder = MiniLMEncoder(minilm_path(assets), max_len=cfg["max_seq_length"])
    X = embeddings if embeddings is not None else encoder.encode([r["text"] for r in train_rows])
    return EmbeddingKNNModel(X, [r["score"] for r in train_rows],
                             [r["question"] for r in train_rows], cfg, encoder)


def encode(texts, assets):
    """Embed texts once (e.g. the whole training set) and slice per fold:
    fit(..., embeddings=E[train_idx]) and model.predict(..., embeddings=E[val_idx])."""
    return MiniLMEncoder(minilm_path(assets)).encode(list(texts))
