"""
The shared state threaded through the ARIA graph.

`failure` is the field that keeps a provider error from ever being mistaken
for an answer. When it is set, `answer` is guaranteed to be empty and every
downstream consumer — the graph's routing, the API layer, the UI — treats
the turn as failed rather than rendering it as clinical content.
"""

from __future__ import annotations

from typing import Any, TypedDict

from llm.errors import AriaLLMError

__all__ = ["AriaFailure", "AriaState", "initial_state", "record_failure"]


class AriaFailure(TypedDict):
    """A provider failure, in a form the API layer can serialise directly."""

    stage: str
    model: str
    code: str | None
    message: str


class AriaState(TypedDict):
    """State object for the compiled LangGraph app."""

    query: str
    chunks: list[Any]
    answer: str
    is_medical: bool
    #: None means "not adjudicated" — never substitute a placeholder number.
    confidence: float | None
    retry_count: int
    #: Set only when an LLM/provider call failed. Mutually exclusive with a
    #: usable `answer`.
    failure: AriaFailure | None
    #: True when the answer is genuine but the Judge could not score it.
    judge_failed: bool


def initial_state(question: str) -> AriaState:
    """A clean state for one consultation."""
    return AriaState(
        query=question,
        chunks=[],
        answer="",
        is_medical=False,
        confidence=None,
        retry_count=0,
        failure=None,
        judge_failed=False,
    )


def record_failure(state: AriaState, exc: AriaLLMError) -> AriaState:
    """Mark the turn as failed and guarantee no answer-shaped text survives."""
    state["failure"] = AriaFailure(
        stage=exc.stage,
        model=exc.model,
        code=exc.code,
        message=exc.public_message(),
    )
    state["answer"] = ""
    state["confidence"] = None
    return state
