import random

from prompt_repetition.tasks import make_middle_match_example, make_name_index_example


def test_name_index_answer_matches_ith_name():
    rng = random.Random(0)
    for _ in range(20):
        ex = make_name_index_example(rng, n=50, i=25)
        names = ex.extra["names"]
        assert len(names) == 50
        assert ex.answer == names[24]
        assert "25th name" in ex.query
        assert ex.answer in ex.query


def test_middle_match_answer_is_unique_and_correct():
    rng = random.Random(1)
    for _ in range(20):
        ex = make_middle_match_example(rng, n=40, k=10)
        seq = ex.extra["seq"]
        x, y = ex.extra["x"], ex.extra["y"]
        assert len(seq) == 40

        matches = [
            j
            for j in range(1, len(seq) - 1)
            if {seq[j - 1], seq[j + 1]} == {x, y}
        ]
        assert len(matches) == 1
        assert seq[matches[0]] == ex.answer
        assert f"between {x} and {y}" in ex.query


def test_middle_match_is_reproducible_with_same_seed():
    ex1 = make_middle_match_example(random.Random(7), n=40, k=10)
    ex2 = make_middle_match_example(random.Random(7), n=40, k=10)
    assert ex1.query == ex2.query
    assert ex1.answer == ex2.answer
