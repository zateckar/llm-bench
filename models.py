"""Shared data structures for the LLM benchmark."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Question:
    id: str
    category: str
    prompt: str
    evaluator: str
    expected: Any = None
    system_prompt: str | None = None


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Result:
    question: Question
    response: str
    score: float
    detail: str = ""
    tokens: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class CategoryResult:
    name: str
    results: list[Result] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.score >= 0.5)

    @property
    def score_pct(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results) * 100

    @property
    def tokens(self) -> TokenUsage:
        total_prompt = sum(r.tokens.prompt_tokens for r in self.results)
        total_completion = sum(r.tokens.completion_tokens for r in self.results)
        return TokenUsage(prompt_tokens=total_prompt, completion_tokens=total_completion)
