"""Build the Approach 2 artifacts: embed every training response with all-MiniLM-L6-v2 and save
the embeddings, Score1 grades, question ids and the CV-selected KNN settings.

Usage (from the repo root):
    python tuning/approach2/train_approach2.py \
        --data data/training_dataset.csv \
        --minilm local_assets/minilm \
        --out submission/weights/approach2

--minilm is the encoder folder from `python create_environment.py --download-encoders`
(its path is also listed in local_assets/assets.json under models.minilm.path).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission"))
from approach2 import MiniLMEncoder  # same encoder code the grader runs

# KNN settings saved with the weights (override after the team CV run)
CONFIG = {
    "encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "encoder_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "max_seq_length": 256,
    "center": True,
    "large_min": 100,
    "k_large": 11, "combine_large": "vote",
    "k_small": 3, "combine_small": "wmean",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "training_dataset.csv"))
    ap.add_argument("--minilm", required=True, help="path to the all-MiniLM-L6-v2 folder")
    ap.add_argument("--out", default=str(ROOT / "submission" / "weights" / "approach2"))
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    X = MiniLMEncoder(a.minilm).encode(df["EssayText"].fillna("").astype(str).tolist())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "train_embeddings.npz",
                        embeddings=X.astype(np.float16),
                        grades=df["Score1"].astype(np.float32).values,
                        questions=np.asarray(df["EssaySet"].astype(str).tolist(), dtype="U"))
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    print(f"saved {X.shape} embeddings for {df['EssaySet'].nunique()} questions to {out}")


if __name__ == "__main__":
    main()
