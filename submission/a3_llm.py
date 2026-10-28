"""Approach 3, LLM stage: grade answers to the small (0-5) questions with a Purdue GenAI Studio
model, using that question's graded training answers as examples.

The question text is not available (questions.json has an empty "context"), so the model is shown
the instructor-graded answers and asked to infer the question and the grading standard from them,
then grade new answers on the same scale. Answers are graded in batches (one call per question
when possible) to stay far below the 150-request budget.

Safety: hard time budget, hard request budget, strict output validation. Anything that fails,
times out or cannot be parsed simply keeps the non-LLM model's grade, so this stage can only
replace predictions, never break the submission. Uses only the course wrapper (genai_wrapper.py)
and the student's key in credentials.py; the key is never printed.
"""
import json
import re
import threading
import time

DEFAULTS = {
    "enabled": True,
    "model": "gpt-oss:120b",
    "weight": 0.7,             # final = weight * LLM grade + (1 - weight) * model grade
    "chunk": 30,               # max new answers per request
    "max_workers": 6,          # requests in flight (the wrapper paces request starts at ~20/min)
    "time_budget_s": 1200,     # stop waiting for the LLM after this many seconds
    "max_requests": 135,       # hard cap on requests (grader budget is 150, incl. retries)
    "max_tokens": 6000,        # room for reasoning models; the answer itself is short
    "max_words_example": 70,   # truncate very long example answers
    "max_words_answer": 120,
    "temperature": 0,
}

SYSTEM = ("You are an experienced computer science instructor grading students' short answers. "
          "You grade consistently with the instructor's examples.")

PROMPT = """Below are students' answers to ONE short-answer exam question, each with the grade the instructor gave.
Grades go from {lo} to {hi} in steps of {step}. The question text is not shown: work out from the graded examples what the question asks and what earns full credit, partial credit or no credit.

GRADED EXAMPLES (instructor grade in brackets):
{examples}

Now grade the following NEW answers to the same question, with exactly the same standard as the instructor. Similar answers must get similar grades. A correct answer in different words deserves the same credit.

NEW ANSWERS:
{answers}

Reply with ONLY a JSON object that maps every new answer's id to its grade, for example {{{example_json}}}. No other text."""


def _clean(s, max_words):
    s = re.sub(r"<[^>]+>", " ", "" if s is None else str(s))
    w = s.split()
    return (" ".join(w[:max_words]) + (" ..." if len(w) > max_words else "")) or "(blank)"


def _fmt(g):
    g = float(g)
    return str(int(g)) if g == int(g) else f"{g:g}"


def build_prompt(examples, answers, allowed, cfg):
    """examples: list of (text, grade); answers: list of (aid, text)."""
    allowed = sorted(float(a) for a in allowed)
    step = min((b - a for a, b in zip(allowed, allowed[1:])), default=1.0)
    ex = sorted(examples, key=lambda e: -float(e[1]))
    ex_lines = "\n".join(f"[{_fmt(g)}] {_clean(t, cfg['max_words_example'])}" for t, g in ex)
    ans_lines = "\n".join(f"{aid}: {_clean(t, cfg['max_words_answer'])}" for aid, t in answers)
    demo = ", ".join(f'"{aid}": {_fmt(allowed[-1 - (i % 2)])}' for i, (aid, _) in enumerate(answers[:2]))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(lo=_fmt(allowed[0]), hi=_fmt(allowed[-1]), step=_fmt(step),
                                                      examples=ex_lines, answers=ans_lines, example_json=demo)}]


def parse_grades(text, ids, allowed):
    """Return {aid: grade} for every id we can read and validate; ignores anything else."""
    if not isinstance(text, str):
        return {}
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    lo, hi = min(allowed), max(allowed)
    out = {}
    for blob in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.S)):      # last JSON object wins
        try:
            obj = json.loads(blob)
        except ValueError:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                k = str(k).strip()
                if k in ids and k not in out:
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                    if v == v and lo - 1e-9 <= v <= hi + 1e-9:
                        out[k] = v
        if out:
            break
    if not out:   # fallback: lines like "A3: 4.5"
        for aid, val in re.findall(r"\b(A\d+)\b\"?\s*[:=]\s*\"?(-?\d+(?:\.\d+)?)", text):
            v = float(val)
            if aid in ids and aid not in out and lo - 1e-9 <= v <= hi + 1e-9:
                out[aid] = v
    return out


class LLMGrader:
    """Runs the LLM stage under a time and request budget. Thread-safe request counting."""

    def __init__(self, cfg, client=None, log=print):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.client = client
        self.log = log
        self.requests = 0
        self.failures = 0
        self.unusable = 0
        self.lock = threading.Lock()
        self.dead = False

    def _client(self):
        if self.client is None:
            from credentials import API_KEY           # the student's own key; never printed
            from genai_wrapper import OpenAI
            self.client = OpenAI(api_key=API_KEY)
        return self.client

    def _take_request(self):
        with self.lock:
            if self.dead or self.requests >= self.cfg["max_requests"]:
                return False
            self.requests += 1
            return True

    def _call(self, messages):
        resp = self._client().chat.completions.create(
            model=self.cfg["model"], messages=messages,
            temperature=self.cfg["temperature"], max_tokens=int(self.cfg["max_tokens"]))
        return resp.choices[0].message.content

    def _job(self, examples, answers, allowed, deadline):
        """Grade one batch [(short_id, key, text)]; one retry for failures or missing grades.
        Returns {key: grade}."""
        key_of = {a: k for a, k, _ in answers}
        got = {}
        for attempt in range(2):
            todo = [(a, t) for a, _, t in answers if a not in got]
            if not todo or time.monotonic() > deadline or not self._take_request():
                break
            try:
                text = self._call(build_prompt(examples, todo, allowed, self.cfg))
                new = parse_grades(text, {a for a, _ in todo}, allowed)
                got.update(new)
                if not new:
                    with self.lock:
                        self.unusable += 1
                        if self.unusable >= 10 and self.unusable > 0.5 * self.requests:
                            self.dead = True          # model keeps replying in an unusable format
            except Exception as exc:                  # GenAIError, transport, anything
                with self.lock:
                    self.failures += 1
                    if self.failures >= 8 and self.failures > 2 * max(1, self.requests - self.failures):
                        self.dead = True              # service looks down: stop trying
                name = type(exc).__name__
                self.log(f"[approach3-llm] request failed ({name}); keeping model grades for this batch")
        return {key_of[a]: g for a, g in got.items()}

    def grade(self, tasks):
        """tasks: output of make_tasks(). Returns {key: grade} for every answer that got a valid grade.
        Uses daemon worker threads, so a hung request can never stop the program from finishing."""
        import queue
        cfg = self.cfg
        t0 = time.monotonic()
        deadline = t0 + float(cfg["time_budget_s"])
        results, done = {}, [0]
        q = queue.Queue()
        for t in tasks:
            q.put(t)

        def worker():
            while time.monotonic() < deadline and not self.dead:
                try:
                    ex, ans, al = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    got = self._job(ex, ans, al, deadline)
                except Exception:
                    got = {}
                with self.lock:
                    results.update(got)
                    done[0] += 1

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(int(cfg["max_workers"]))]
        for th in threads:
            th.start()
        for th in threads:
            th.join(max(0.0, deadline - time.monotonic()))
        with self.lock:
            unfinished = len(tasks) - done[0]
            self.dead = True                     # nothing new starts after this point
            snapshot = dict(results)
        if unfinished:
            self.log(f"[approach3-llm] stopped early; {unfinished} batches keep model grades")
            try:
                from genai_wrapper import get_client
                get_client().cancel()
            except Exception:
                pass
        self.log(f"[approach3-llm] {len(snapshot)} answers graded by {cfg['model']} with {self.requests} requests "
                 f"({self.failures} failed) in {time.monotonic() - t0:.0f}s")
        return snapshot


def make_tasks(groups, examples_by_q, allowed_by_q, chunk):
    """groups: {qid: [(key, text), ...]} with any hashable key per answer.
    Returns batches (examples, [(short_id, key, text)], allowed): each question is split into
    batches of at most `chunk` answers; the first batch of every question comes first, so under a
    tight budget every question gets at least some LLM grades."""
    per_q = []
    for qid, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ex = examples_by_q.get(qid)
        if not ex or not items:
            continue
        n = max(1, -(-len(items) // chunk))
        size = -(-len(items) // n)
        batches = []
        for i in range(0, len(items), size):
            part = items[i:i + size]
            batches.append((ex, [(f"A{j + 1}", k, t) for j, (k, t) in enumerate(part)], allowed_by_q[qid]))
        per_q.append(batches)
    ordered = []
    for r in range(max((len(b) for b in per_q), default=0)):
        ordered += [b[r] for b in per_q if r < len(b)]
    return ordered
