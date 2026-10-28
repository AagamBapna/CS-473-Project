"""Train Approach 3 on graded answers and grade a new file.

  python predict.py --test path/to/test.csv                   # uses data/train.csv
  python predict.py --train other_train.csv --test test.csv --out preds.csv
  python predict.py --selfcheck                               # hold out 20%, train on rest, report QWK

Column names are auto-detected (Kaggle style Id/EssaySet/EssayText/Score1 or the
handout's Question Number/Student Submission/Manual Score). Score ranges are read from
the training data, so new questions and new scales work without code changes.

If Colab outputs exist (outputs/oof_tx_*.csv + matching test_tx_*.csv, emb_*.npz covering
the test ids, llm_feats.csv with context 'full'), they are used automatically; otherwise
the CPU-only stack runs on its own.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import OUT, load, qwk
from stack import fit_predict

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(HERE, "data", "train.csv"))
    ap.add_argument("--test")
    ap.add_argument("--out", default=os.path.join(OUT, "predictions.csv"))
    ap.add_argument("--stacker", default="blend", choices=["ridge", "hgb", "blend"])
    ap.add_argument("--metric", default="pooled", choices=["pooled", "perq"])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()

    train = load(a.train)
    t0 = time.time()
    if a.selfcheck:
        tr, te = train_test_split(train, test_size=0.2, random_state=7,
                                  stratify=train.q.astype(str) + "_" + train.y.astype(str))
        truth = te.y.values
        te = te.drop(columns=[c for c in ("y", "y2") if c in te])
        # external CV files would leak here (their OOF preds came from other splits) -> CPU stack only
        res = fit_predict(tr, te, a.stacker, n_jobs=a.jobs, ext={"emb": {}, "cols": {}, "llm": None}, metric=a.metric)
        res = res.set_index("id").loc[te.id]
        print(f"self-check on 20% held-out answers: pooled QWK {qwk(truth, res.pred.values):.4f}, "
              f"mean per-question QWK {np.mean([qwk(truth[te.q.values == q], res.pred.values[te.q.values == q]) for q in te.q.unique()]):.4f}")
    else:
        if not a.test:
            ap.error("--test is required (or use --selfcheck)")
        test = load(a.test)
        res = fit_predict(train, test, a.stacker, n_jobs=a.jobs, metric=a.metric)
        raw = pd.read_csv(a.test)
        id_col = next(c for c in raw.columns if c.lower() in ("id", "submission id", "submission_id"))
        out = pd.DataFrame({id_col: res.id, "PredictedScore": res.pred})
        out.to_csv(a.out, index=False)
        print(f"wrote {len(out)} predictions to {a.out}")
        print(out.PredictedScore.value_counts().sort_index().to_string())
    print(f"done in {time.time() - t0:.0f}s")
