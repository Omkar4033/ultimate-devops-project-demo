# Prompt Repetition Benchmark

A small, self-contained reproduction of the prompting variants and custom
benchmark tasks from **"Prompt Repetition Improves Non-Reasoning LLMs"**
(arXiv:[2512.14982](https://arxiv.org/abs/2512.14982), Leviathan, Kalman,
Matias — Google Research), specifically:

- **Appendix A.4** — the four prompting variants (Baseline, Prompt
  Repetition, Prompt Repetition (Verbose), Prompt Repetition ×3).
- **Appendix A.3** — the two custom benchmark tasks (NameIndex,
  MiddleMatch) the paper highlights as showing the strongest gains from
  prompt repetition.

Run it yourself against a real model to see the accuracy gap the paper
reports (e.g. it cites Gemini 2.0 Flash-Lite on NameIndex going from
21.33% baseline to 97.33% with prompt repetition).

## The four prompting variants (Appendix A.4)

Given a raw query `<QUERY>`, each method sends a different transformed
prompt to the model (transcribed from the paper's "Query Examples" table):

| Method | Prompt sent to the model |
|---|---|
| `baseline` | `<QUERY>` |
| `repeat` (Prompt Repetition) | `<QUERY>` `<QUERY>` |
| `repeat_verbose` (Verbose) | `<QUERY>` "Let me repeat that:" `<QUERY>` |
| `repeat_x3` (×3) | `<QUERY>` "Let me repeat that:" `<QUERY>` "Let me repeat that one more time:" `<QUERY>` |

Implemented verbatim in [`prompt_repetition/templates.py`](prompt_repetition/templates.py).
No reasoning/thinking is enabled for any provider — this is the
"non-reasoning" setting the paper studies (prompt repetition is reported
as neutral-to-slightly-positive, not the headline result, when reasoning
is turned on).

## The two benchmark tasks (Appendix A.3)

- **NameIndex** (`nameindex`): the model is given a list of `N=50` random
  names and asked for the `i=25`-th name. Trivial for a program, hard for
  an LLM that can't "look ahead" at the full list while forming its answer
  — exactly the causal-attention asymmetry the paper's repetition trick
  targets.
- **MiddleMatch** (`middlematch`): the model is given a list of `N=40`
  names drawn (with repetition) from a pool of only `K=10` distinct names,
  and asked for the single name that sits directly between two named
  occurrences. The generator ([`prompt_repetition/tasks.py`](prompt_repetition/tasks.py))
  guarantees the queried pair has exactly one valid "middle" position, so
  each example has one unambiguous correct answer — mirroring the paper's
  worked example.

Both generators are seeded (`--seed`) for reproducibility, and each
example asks the model to answer in a fixed `The answer is <ANSWER>.`
format so responses can be graded automatically
([`prompt_repetition/grading.py`](prompt_repetition/grading.py)); the paper's
own printed examples for these two tasks don't show this instruction, so
it's a small necessary addition for automated scoring, not part of the
task logic itself.

## Setup

```bash
cd prompt-repetition-benchmark
python3 -m pip install -r requirements.txt   # only need the provider(s) you'll use
```

You only need `anthropic` if using `--provider anthropic`, or `openai` if
using `--provider openai`. Set the corresponding API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Offline smoke test of the whole pipeline -- no API key, no network calls.
# Uses a synthetic mock "model" (see NOTE below); good for verifying the
# harness works before spending real API credits.
python3 run.py --provider mock

# Real reproduction against Claude:
python3 run.py --provider anthropic --model claude-haiku-4-5-20251001 --num-examples 30

# Real reproduction against GPT, one task only:
python3 run.py --provider openai --model gpt-4o-mini --task nameindex --num-examples 30

# Only compare baseline vs. plain prompt repetition:
python3 run.py --provider anthropic --methods baseline,repeat

# Save machine-readable results:
python3 run.py --provider anthropic --output results.json
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--task` | `all` | `nameindex`, `middlematch`, or `all` |
| `--provider` | `mock` | `anthropic`, `openai`, or `mock` |
| `--model` | provider default | model id/name |
| `--num-examples` | `20` | examples generated per method per task |
| `--methods` | all four | comma-separated subset of `baseline,repeat,repeat_verbose,repeat_x3` |
| `--concurrency` | `4` | parallel API calls |
| `--seed` | `42` | RNG seed for example generation |
| `--verbose` | off | print each example's expected/actual answer |
| `--output` | none | path to write a JSON results file |

Example output:

```
=== nameindex (provider=anthropic, model=claude-haiku-4-5-20251001, n=30/method) ===
Baseline                       30.00%  (  9/30 )  ############
Prompt Repetition               76.67%  ( 23/30 )  ###############################
Prompt Repetition (Verbose)     80.00%  ( 24/30 )  ################################
Prompt Repetition x3            90.00%  ( 27/30 )  ####################################
```

(Illustrative only — actual numbers depend on the model you run.)

### About `--provider mock`

`mock` is a fully offline, synthetic stand-in with zero network calls. It
parses the generated NameIndex/MiddleMatch prompt just enough to know the
correct answer, then "guesses" correctly with a probability that increases
with how many times the query was repeated in the prompt it received —
a toy simulation used only to verify the harness (template formatting,
task generation, grading, aggregation) end-to-end. It is **not** a real
language model and its numbers say nothing about real LLM behavior — use
`anthropic` or `openai` for that.

## Project layout

```
prompt-repetition-benchmark/
  run.py                        CLI entry point
  prompt_repetition/
    templates.py                the 4 prompting variants (Appendix A.4)
    tasks.py                    NameIndex / MiddleMatch generators (Appendix A.3)
    names_data.py                first/last name pools used to synthesize examples
    grading.py                   answer extraction + correctness check
    llm_client.py                Anthropic / OpenAI / mock API clients
    runner.py                    orchestrates a benchmark run (threaded)
  tests/                        unit tests (templates, tasks, grading)
```

## Tests

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
```

Tests only cover the offline logic (template formatting, task generation
correctness/uniqueness, grading) — they don't call any real API.
