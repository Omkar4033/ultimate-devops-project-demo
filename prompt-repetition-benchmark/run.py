#!/usr/bin/env python3
"""CLI to reproduce the NameIndex / MiddleMatch prompt-repetition benchmarks
from arXiv:2512.14982 ("Prompt Repetition Improves Non-Reasoning LLMs"),
Appendix A.3 (tasks) and Appendix A.4 (prompting variants).

Examples:
    # Offline smoke test of the whole pipeline, no API key needed:
    python run.py --provider mock

    # Real reproduction against Claude:
    export ANTHROPIC_API_KEY=...
    python run.py --provider anthropic --model claude-haiku-4-5-20251001 --num-examples 30

    # Real reproduction against GPT:
    export OPENAI_API_KEY=...
    python run.py --provider openai --model gpt-4o-mini --task nameindex
"""

from __future__ import annotations

import argparse
import json
import sys

from prompt_repetition.llm_client import build_client
from prompt_repetition.runner import run_benchmark
from prompt_repetition.templates import METHOD_LABELS, METHODS

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "mock": "mock-demo",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["nameindex", "middlematch", "all"], default="all")
    parser.add_argument("--provider", choices=["anthropic", "openai", "mock"], default="mock")
    parser.add_argument("--model", default=None, help="Model id. Defaults per-provider (see DEFAULT_MODELS).")
    parser.add_argument("--api-key", default=None, help="Override API key; else read from env var.")
    parser.add_argument("--num-examples", type=int, default=20, help="Examples per method per task.")
    parser.add_argument(
        "--methods",
        default="baseline,repeat,repeat_verbose,repeat_x3",
        help=f"Comma-separated subset of: {', '.join(METHODS)}",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--verbose", action="store_true", help="Print per-example outcomes.")
    parser.add_argument("--output", default=None, help="Optional path to save JSON results.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            print(f"error: unknown method {m!r}. choose from: {', '.join(METHODS)}", file=sys.stderr)
            return 2

    model = args.model or DEFAULT_MODELS[args.provider]
    try:
        client = build_client(args.provider, model, api_key=args.api_key)
    except Exception as e:
        print(f"error: could not initialize provider {args.provider!r}: {e}", file=sys.stderr)
        return 1

    tasks = ["nameindex", "middlematch"] if args.task == "all" else [args.task]

    all_results = {}
    for task in tasks:
        print(f"\n=== {task} (provider={args.provider}, model={model}, n={args.num_examples}/method) ===")
        summary, _ = run_benchmark(
            client,
            task,
            methods,
            args.num_examples,
            seed=args.seed,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        all_results[task] = summary

        width = max(len(METHOD_LABELS[m]) for m in methods)
        for m in methods:
            acc = summary[m]["accuracy"]
            bar = "#" * round(acc * 40)
            print(f"{METHOD_LABELS[m]:<{width}}  {acc * 100:6.2f}%  ({summary[m]['correct']:>3}/{summary[m]['n']:<3})  {bar}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    if args.provider == "mock":
        print(
            "\nNOTE: --provider mock is a synthetic, offline stand-in for pipeline testing.\n"
            "It never calls a real model. Use --provider anthropic|openai with a real API key\n"
            "(ANTHROPIC_API_KEY / OPENAI_API_KEY) to reproduce the paper's actual findings."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
