"""The four prompting variants from Appendix A.4 of the paper.

Each function takes a raw query string (<QUERY>) and returns the transformed
prompt to actually send to the model. Transcribed directly from the "Query
Examples" table in Appendix A.4:

  Baseline                  <QUERY>
  Prompt Repetition         <QUERY> <QUERY>
  Prompt Repetition         <QUERY> "Let me repeat that:" <QUERY>
    (Verbose)
  Prompt Repetition x3      <QUERY> "Let me repeat that:" <QUERY>
                             "Let me repeat that one more time:" <QUERY>
"""

from __future__ import annotations


def baseline(query: str) -> str:
    return query


def prompt_repetition(query: str) -> str:
    return f"{query}\n\n{query}"


def prompt_repetition_verbose(query: str) -> str:
    return f"{query}\n\nLet me repeat that:\n\n{query}"


def prompt_repetition_x3(query: str) -> str:
    return (
        f"{query}\n\n"
        "Let me repeat that:\n\n"
        f"{query}\n\n"
        "Let me repeat that one more time:\n\n"
        f"{query}"
    )


METHODS = {
    "baseline": baseline,
    "repeat": prompt_repetition,
    "repeat_verbose": prompt_repetition_verbose,
    "repeat_x3": prompt_repetition_x3,
}

METHOD_LABELS = {
    "baseline": "Baseline",
    "repeat": "Prompt Repetition",
    "repeat_verbose": "Prompt Repetition (Verbose)",
    "repeat_x3": "Prompt Repetition x3",
}
