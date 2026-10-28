"""Measure how well the Purdue GenAI LLM grades the 81 small questions BEFORE using Gradescope.

For each fold, every small question's answers in that fold are hidden; the LLM sees the
question's other graded answers as examples and grades the hidden ones (exactly what Approach 3
does on Gradescope). Then the script compares, per question (QWK, averaged like the leaderboard):
    current model alone  |  LLM alone  |  blends of the two
and prints the best blend weight. Results are saved after every fold, so you can stop with
Ctrl+C and re-run later: finished work is never repeated (no wasted API calls).

  python llm_eval.py --quick                 # 3 calls: checks your key and the model work
  python llm_eval.py                         # all 5 folds (~405 calls, ~30-40 min)
  python llm_eval.py --folds 0 1             # 2 folds (~160 calls, ~15 min)
  python llm_eval.py --model qwen3:30b       # try another model (separate results file)
  python llm_eval.py --folds 0 1 --prompt 2  # improved prompt (separate results file)
  python llm_eval.py --summary               # just print the table from saved results

Needs: credentials.py (your key), genai_wrapper.py, a3_llm.py, oof_model.csv in this folder,
and training_dataset_updated.csv (path via --csv). Your key is never printed or saved.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a3_llm  # noqa: E402

ALLOWED = [x / 2 for x in range(11)]          # 0, 0.5, ..., 5 (questions 11-91)


def qwk(a, b, n):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.bincount(a * n + b, minlength=n * n).reshape(n, n).astype(float)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    W = (i - j) ** 2 / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return 1 - (W * O).sum() / den if den > 0 else np.nan


def snap(x):
    return np.clip(np.round(np.asarray(x, float) * 2) / 2, 0, 5)


def per_question(df, pred):
    vals = []
    for q, d in df.groupby("question"):
        y = np.rint(d.grade.values * 2).astype(int)
        if len(set(y)) < 2:
            continue
        vals.append(qwk(y, np.rint(snap(pred[d.index]) * 2).astype(int), 11))
    return np.nanmean(vals), len(vals)


def summary(res_path, oof):
    if not os.path.exists(res_path):
        print("No results yet.")
        return
    r = pd.read_csv(res_path)
    m = oof.merge(r[["id", "llm_grade"]], on="id", how="inner").reset_index(drop=True)
    folds = [int(x) for x in sorted(m.fold.unique())]
    print(f"\nAnswers graded by the LLM: {len(m)} of {len(oof[oof.fold.isin(folds)])} in folds {folds}")
    base, n = per_question(m, m.model_grade.values)
    print(f"Mean per-question QWK over {n} questions (same answers for every row):")
    print(f"  current model alone       {base:.4f}")
    best = (base, 0.0)
    for w in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        v, _ = per_question(m, w * m.llm_grade.values + (1 - w) * m.model_grade.values)
        label = "LLM alone" if w == 1.0 else f"blend weight {w:.1f}"
        print(f"  {label:25s} {v:.4f}")
        if v > best[0]:
            best = (v, w)
    print(f"\nBEST: weight {best[1]:.1f} -> {best[0]:.4f}  (current model: {base:.4f}, gain {best[0] - base:+.4f})")
    print("Send this table (or the results CSV) back; it contains no key.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training_dataset_updated.csv")
    ap.add_argument("--model", default=a3_llm.DEFAULTS["model"])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prompt", type=int, default=1, choices=[1, 2], help="prompt version (2 = improved)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    oof = pd.read_csv(os.path.join(HERE, "oof_model.csv"))
    tag = a.model.replace(':', '_').replace('/', '_') + ("" if a.prompt == 1 else f"_p{a.prompt}")
    res_path = os.path.join(HERE, f"llm_eval_results_{tag}.csv")
    if a.summary:
        return summary(res_path, oof)

    raw = pd.read_csv(a.csv)
    text = dict(zip(raw.Id, raw.EssayText.astype(str)))
    oof["text"] = oof.id.map(text)
    if oof.text.isna().any():
        raise SystemExit("The CSV does not match oof_model.csv; use training_dataset_updated.csv")

    from credentials import API_KEY
    if not isinstance(API_KEY, str) or len(API_KEY.strip()) < 12 or "PASTE" in API_KEY.upper():
        raise SystemExit("Put your GenAI Studio API key in credentials.py first (API_KEY = \"...\").")

    done = set(pd.read_csv(res_path).id) if os.path.exists(res_path) else set()
    cfg = {**a3_llm.DEFAULTS, "model": a.model, "max_workers": a.workers, "prompt_version": a.prompt,
           "time_budget_s": 10 ** 6, "max_requests": 10 ** 6}

    folds = [0] if a.quick else a.folds
    for f in folds:
        part = oof[(oof.fold == f) & ~oof.id.isin(done)]
        if a.quick:
            part = part[part.question.isin(sorted(part.question.unique())[:3])]
        if part.empty:
            print(f"fold {f}: already done")
            continue
        groups, examples, allowed = {}, {}, {}
        for q, d in part.groupby("question"):
            ex = oof[(oof.question == q) & (oof.fold != f)]
            examples[q] = [[t, g] for t, g in zip(ex.text, ex.grade)]
            groups[q] = [(i, t) for i, t in zip(d.id, d.text)]
            allowed[q] = ALLOWED
        tasks = a3_llm.make_tasks(groups, examples, allowed, cfg["chunk"])
        t0 = time.time()
        print(f"fold {f}: {len(tasks)} requests with {a.model} ...", flush=True)
        got = a3_llm.LLMGrader(cfg).grade(tasks)
        out = pd.DataFrame({"id": list(got), "llm_grade": list(got.values())})
        out["fold"] = f
        out.to_csv(res_path, mode="a", header=not os.path.exists(res_path), index=False)
        print(f"fold {f}: {len(got)}/{len(part)} answers graded in {time.time() - t0:.0f}s (saved)", flush=True)
        if a.quick:
            sample = part[part.id.isin(got)].head(8)
            for _, r in sample.iterrows():
                print(f"   teacher {r.grade:>4}  LLM {got[r.id]:>4}  model {r.model_grade:5.2f} | {r.text[:70]}")
            return
    summary(res_path, oof)


if __name__ == "__main__":
    main()
