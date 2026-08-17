from prompt_repetition.grading import extract_answer, is_correct


def test_extract_answer_standard_format():
    assert extract_answer("The answer is Dale Lopez.") == "dale lopez"


def test_extract_answer_with_quotes_and_asterisks():
    assert extract_answer('The answer is: **"Dale Lopez"**') == "dale lopez"


def test_extract_answer_fallback_last_line():
    assert extract_answer("Sure, here it is:\nDale Lopez") == "dale lopez"


def test_extract_answer_empty():
    assert extract_answer("") == ""
    assert extract_answer(None) == ""


def test_is_correct_exact_match():
    assert is_correct("The answer is Dale Lopez.", "Dale Lopez")


def test_is_correct_case_and_punctuation_insensitive():
    assert is_correct("the answer is: dale lopez!!", "Dale Lopez")


def test_is_correct_wrong_answer():
    assert not is_correct("The answer is Peter Sanchez.", "Dale Lopez")


def test_is_correct_no_answer():
    assert not is_correct("I'm not sure.", "Dale Lopez")
