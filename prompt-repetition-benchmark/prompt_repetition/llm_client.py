"""Pluggable LLM API clients.

Real reproduction of the paper's results needs a real provider (Anthropic
or OpenAI) and a valid API key. `MockClient` is an offline, synthetic
stand-in with no network calls, useful only for smoke-testing the harness
(templates, task generation, grading, aggregation) without spending any
API credits. It does NOT call a real model and its "accuracy" numbers are
not evidence of anything about real LLM behavior.
"""

from __future__ import annotations

import os
import random
import re
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    model: str

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 30) -> str:
        """Returns the model's text response to `prompt`. Reasoning/thinking
        must stay disabled so this measures the non-reasoning setting the
        paper studies."""


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: Optional[str] = None):
        import anthropic  # local import: optional dependency

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, *, max_tokens: int = 30) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: Optional[str] = None):
        import openai  # local import: optional dependency

        self.model = model
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompt: str, *, max_tokens: int = 30) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class MockClient(LLMClient):
    """Offline synthetic client for pipeline smoke-testing (see module docstring).

    It parses the (synthetic) NameIndex/MiddleMatch prompt it's given well
    enough to know the correct answer, then reports it correctly with a
    probability ("skill") that increases with how many times the query was
    repeated in the prompt -- a toy stand-in for the effect the paper
    measures with real models. This lets `--provider mock` demonstrate the
    full pipeline (including an accuracy gap across methods) with zero API
    calls; it is not a substitute for running against a real model.
    """

    def __init__(self, model: str = "mock-demo", seed: int = 0):
        self.model = model
        self._rng = random.Random(seed)

    def generate(self, prompt: str, *, max_tokens: int = 30) -> str:
        names = self._parse_names(prompt)
        correct_answer = self._solve(prompt, names) if names else None
        skill = self._estimate_skill(prompt)

        if correct_answer is not None and self._rng.random() < skill:
            answer = correct_answer
        elif names:
            candidates = [n for n in names if n != correct_answer] or names
            answer = self._rng.choice(candidates)
        else:
            answer = "Unknown"

        return f"The answer is {answer}."

    @staticmethod
    def _parse_names(prompt: str):
        match = re.search(r"Here's a list.*?:\n\n(.*?)\n\nWhat", prompt, re.DOTALL)
        if not match:
            return None
        return [n.strip() for n in match.group(1).split(",")]

    @staticmethod
    def _solve(prompt: str, names):
        m = re.search(r"What's the (\d+)(?:st|nd|rd|th) name", prompt)
        if m:
            idx = int(m.group(1))
            return names[idx - 1] if 1 <= idx <= len(names) else None

        m = re.search(r"right between (.+?) and (.+?)\?", prompt)
        if m:
            x, y = m.group(1).strip(), m.group(2).strip()
            for j in range(1, len(names) - 1):
                if {names[j - 1], names[j + 1]} == {x, y}:
                    return names[j]
        return None

    @staticmethod
    def _estimate_skill(prompt: str) -> float:
        repeats = prompt.count("Here's a list")
        verbose_bonus = prompt.lower().count("let me repeat")
        skill = 0.25 + 0.22 * (repeats - 1) + 0.05 * verbose_bonus
        return min(0.95, max(0.05, skill))


def build_client(provider: str, model: str, api_key: Optional[str] = None) -> LLMClient:
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key)
    if provider == "openai":
        return OpenAIClient(model=model, api_key=api_key)
    if provider == "mock":
        return MockClient(model=model)
    raise ValueError(f"Unknown provider: {provider!r} (choose: anthropic, openai, mock)")
