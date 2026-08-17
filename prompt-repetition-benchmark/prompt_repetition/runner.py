"""Runs a benchmark task across prompting methods and aggregates accuracy."""

from __future__ import annotations

import concurrent.futures as cf
import random
import time
from typing import Dict, List, Tuple

from .grading import is_correct
from .llm_client import LLMClient
from .tasks import Example, TASKS
from .templates import METHODS


def _generate_with_retry(client: LLMClient, prompt: str, max_tokens: int, retries: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return client.generate(prompt, max_tokens=max_tokens)
        except Exception as e:  # provider/network errors: retry with backoff
            last_error = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return f"[ERROR: {last_error}]"


def run_benchmark(
    client: LLMClient,
    task_name: str,
    methods: List[str],
    num_examples: int,
    seed: int = 42,
    concurrency: int = 4,
    max_tokens: int = 30,
    verbose: bool = False,
) -> Tuple[Dict[str, dict], List[Example]]:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name!r} (choose: {', '.join(TASKS)})")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method!r} (choose: {', '.join(METHODS)})")

    rng = random.Random(seed)
    task_fn = TASKS[task_name]
    examples = [task_fn(rng) for _ in range(num_examples)]

    jobs = []
    for method in methods:
        template_fn = METHODS[method]
        for idx, ex in enumerate(examples):
            jobs.append((method, idx, ex, template_fn(ex.query)))

    outcomes: Dict[str, List[bool]] = {method: [False] * num_examples for method in methods}
    responses: Dict[str, List[str]] = {method: [""] * num_examples for method in methods}

    def _worker(job):
        method, idx, ex, prompt = job
        response = _generate_with_retry(client, prompt, max_tokens)
        correct = is_correct(response, ex.answer)
        return method, idx, correct, response

    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for method, idx, correct, response in pool.map(_worker, jobs):
            outcomes[method][idx] = correct
            responses[method][idx] = response
            if verbose:
                mark = "OK " if correct else "X  "
                print(f"  [{task_name}] {method:<15} #{idx:<3} {mark} expected={examples[idx].answer!r} got={response!r}")

    summary = {}
    for method in methods:
        n_correct = sum(1 for o in outcomes[method] if o)
        summary[method] = {
            "accuracy": n_correct / num_examples,
            "correct": n_correct,
            "n": num_examples,
        }
    return summary, examples
