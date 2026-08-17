from prompt_repetition.templates import (
    baseline,
    prompt_repetition,
    prompt_repetition_verbose,
    prompt_repetition_x3,
)


def test_baseline_is_identity():
    assert baseline("Q") == "Q"


def test_prompt_repetition_repeats_twice():
    out = prompt_repetition("Q")
    assert out.count("Q") == 2
    assert "Let me repeat" not in out


def test_prompt_repetition_verbose_has_marker_once():
    out = prompt_repetition_verbose("Q")
    assert out.count("Q") == 2
    assert out.count("Let me repeat that:") == 1
    assert "one more time" not in out
    # query must appear before and after the marker
    marker_pos = out.index("Let me repeat that:")
    assert out.index("Q") < marker_pos < out.rindex("Q")


def test_prompt_repetition_x3_repeats_three_times():
    out = prompt_repetition_x3("Q")
    assert out.count("Q") == 3
    assert out.count("Let me repeat that:") == 1
    assert out.count("Let me repeat that one more time:") == 1
    # ordering: Q, marker1, Q, marker2, Q
    i1 = out.index("Q")
    m1 = out.index("Let me repeat that:")
    i2 = out.index("Q", m1)
    m2 = out.index("Let me repeat that one more time:")
    i3 = out.index("Q", m2)
    assert i1 < m1 < i2 < m2 < i3
