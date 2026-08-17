"""NameIndex and MiddleMatch benchmark tasks from Appendix A.3 of the paper.

NameIndex: the model gets a list of N names and must output the i-th name
on the list. Paper uses N=50, i=25.

MiddleMatch: the model gets a list of N names (drawn, with repetition, from
a pool of K < N possible names) and must output the single name located
directly between two given ones. Paper uses N=40, K=10.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .names_data import FIRST_NAMES, LAST_NAMES

DEFAULT_NAME_INDEX_N = 50
DEFAULT_NAME_INDEX_I = 25
DEFAULT_MIDDLE_MATCH_N = 40
DEFAULT_MIDDLE_MATCH_K = 10

ANSWER_FORMAT_INSTRUCTION = (
    "Reply with just the name, in the format: The answer is <ANSWER>."
)


@dataclass
class Example:
    task: str
    query: str
    answer: str
    extra: Dict[str, Any] = field(default_factory=dict)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def random_full_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def make_name_index_example(
    rng: random.Random,
    n: int = DEFAULT_NAME_INDEX_N,
    i: int = DEFAULT_NAME_INDEX_I,
) -> Example:
    names = [random_full_name(rng) for _ in range(n)]
    answer = names[i - 1]
    query = (
        "Here's a list of names:\n\n"
        f"{', '.join(names)}\n\n"
        f"What's the {_ordinal(i)} name?\n\n"
        f"{ANSWER_FORMAT_INSTRUCTION}"
    )
    return Example(task="NameIndex", query=query, answer=answer, extra={"names": names, "i": i})


def make_middle_match_example(
    rng: random.Random,
    n: int = DEFAULT_MIDDLE_MATCH_N,
    k: int = DEFAULT_MIDDLE_MATCH_K,
    max_attempts: int = 500,
) -> Example:
    """Generates a list with repetitions such that exactly one pair of
    positions (j-1, j+1) holds two distinct chosen query names, so the
    "name directly between X and Y" question has a unique, well-defined
    answer (mirroring the worked example in Appendix A.3).
    """
    pool = [random_full_name(rng) for _ in range(k)]

    for _ in range(max_attempts):
        seq = [rng.choice(pool) for _ in range(n)]

        pair_positions: Dict[frozenset, List[int]] = {}
        for j in range(1, n - 1):
            a, b = seq[j - 1], seq[j + 1]
            if a == b:
                continue
            key = frozenset((a, b))
            pair_positions.setdefault(key, []).append(j)

        unique_pairs = [(key, positions[0]) for key, positions in pair_positions.items() if len(positions) == 1]
        if not unique_pairs:
            continue

        key, j = rng.choice(unique_pairs)
        x, y = tuple(key)
        if rng.random() < 0.5:
            x, y = y, x
        answer = seq[j]

        query = (
            "Here's a list (potentially with repetitions) of names:\n\n"
            f"{', '.join(seq)}\n\n"
            f"What is the single name that appears right between {x} and {y}?\n\n"
            f"{ANSWER_FORMAT_INSTRUCTION}"
        )
        return Example(
            task="MiddleMatch",
            query=query,
            answer=answer,
            extra={"seq": seq, "x": x, "y": y, "position": j},
        )

    raise RuntimeError(
        "Failed to generate a MiddleMatch example with a unique answer; "
        "try increasing max_attempts or adjusting n/k."
    )


TASKS = {
    "nameindex": lambda rng: make_name_index_example(rng),
    "middlematch": lambda rng: make_middle_match_example(rng),
}
