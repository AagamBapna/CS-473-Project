"""Approach 3 (Open Approach): Rubric-Mined Stacked Grader.

Instead of copying grades from similar answers (Approaches 1-2), this learns what earns
points. For each question, five level-1 graders score an answer:
  1. TF-IDF ridge regression (word 1-2-grams on spell-corrected text + char 2-5-grams)
  2. TF-IDF multinomial logistic regression (expected grade, P(lowest), P(highest))
  3. automatically mined rubric (credit / misconception phrases chosen by z-test)
  4. exemplar similarity (closest training answers at each grade level)
  5. surface statistics (length, vocabulary, misspelling rate, ...)
A ridge stacker combines them (trained on out-of-fold features) and per-question cut points,
tuned to maximise QWK, turn the score into a grade.

Questions with only ~25 graded answers (the 0-5, half-point questions) are too small for that.
They use ONE cross-question model (a3_small.py): each answer is described relative to its own
question's graded answers (similarity to the full-mark answers, nearest graded neighbours,
coverage of the full-mark answers' shared vocabulary, relative length), and a single ridge model
trained on all small questions maps that to a grade.

Trained offline by train_approach3.py; artifacts are plain numpy/JSON files in
weights/approach3/. No GenAI calls. Grades are snapped to questions[q]["allowed_grades"].
"""
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from a3_infer import Approach3Model  # noqa: E402


def _find_weights(weights_dir):
    candidates = []
    if weights_dir:
        candidates += [Path(weights_dir) / "approach3", Path(weights_dir)]
    candidates += [_HERE / "weights" / "approach3"]
    for c in candidates:
        if (c / "index.json").exists():
            return c
    raise FileNotFoundError(f"approach3 artifacts not found in {[str(c) for c in candidates]}")


def load_model(weights_dir, config=None, assets=None):
    # Load saved artifacts only. Nothing is trained here.
    return Approach3Model(_find_weights(weights_dir))
