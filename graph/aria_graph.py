"""
The ARIA graph: guardrail -> navigator -> generator -> judge.

Routing is failure-aware at every hop. Once `state["failure"]` is set the
graph goes straight to END, so a dead model can never be carried forward
into retrieval, generation or scoring.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    generator_node,
    guardrail_node,
    judge_node,
    navigator_node,
    reject_node,
)
from graph.state import AriaState, initial_state

logger = logging.getLogger(__name__)

__all__ = ["AriaState", "ask_aria", "build_aria", "run_aria"]

CONFIDENCE_THRESHOLD = 0.7
MAX_RETRIES = 3


def guardrail_decision(state: AriaState) -> str:
    if state["failure"] is not None:
        return "fail"
    return "navigator" if state["is_medical"] else "reject"


def navigator_decision(state: AriaState) -> str:
    return "fail" if state["failure"] is not None else "generator"


def generator_decision(state: AriaState) -> str:
    return "fail" if state["failure"] is not None else "judge"


def judge_decision(state: AriaState) -> str:
    if state["failure"] is not None:
        return "end"

    confidence = state["confidence"]
    # An unadjudicated answer is returned as-is rather than retried: the
    # Judge is unavailable, so retrying cannot produce a better verdict.
    if confidence is None:
        return "end"

    if confidence >= CONFIDENCE_THRESHOLD:
        return "end"
    if state["retry_count"] >= MAX_RETRIES:
        logger.info("max retries reached — best available answer returned")
        return "end"

    logger.info("retry %d — refining answer", state["retry_count"])
    return "retry"


def build_aria() -> object:
    graph: StateGraph[AriaState, Any, Any, Any] = StateGraph(AriaState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("navigator", navigator_node)
    graph.add_node("generator", generator_node)
    graph.add_node("judge", judge_node)
    graph.add_node("reject", reject_node)

    graph.add_edge(START, "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        guardrail_decision,
        {"navigator": "navigator", "reject": "reject", "fail": END},
    )
    graph.add_conditional_edges(
        "navigator",
        navigator_decision,
        {"generator": "generator", "fail": END},
    )
    graph.add_conditional_edges(
        "generator",
        generator_decision,
        {"judge": "judge", "fail": END},
    )
    graph.add_conditional_edges(
        "judge",
        judge_decision,
        {"retry": "generator", "end": END},
    )
    graph.add_edge("reject", END)

    return graph.compile()


# Compiled once and reused across calls
_app: object | None = None


def run_aria(question: str) -> AriaState:
    """Run one consultation and return the full final state.

    Prefer this over `ask_aria` when the caller needs to distinguish an
    answer from a failure — which is every caller that renders to a user.
    """
    global _app
    if _app is None:
        _app = build_aria()

    final: AriaState = _app.invoke(initial_state(question))  # type: ignore[attr-defined]
    return final


def ask_aria(question: str) -> str:
    """Convenience wrapper returning just the answer text.

    Raises:
        AriaLLMError: if the consultation failed. It deliberately does not
            return the error as a string — that is the bug this refactor
            exists to remove.
    """
    from llm.errors import AriaLLMError

    final = run_aria(question)
    failure = final["failure"]
    if failure is not None:
        raise AriaLLMError(
            stage=failure["stage"],
            model=failure["model"],
            message=failure["message"],
            code=failure["code"],
        )
    return final["answer"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for label, probe in [
        ("Medical Query", "What is the treatment for hypertension?"),
        ("Non-Medical Query", "What is the price of Bitcoin?"),
    ]:
        print("=" * 60)
        print(f"TEST: {label}")
        print("=" * 60)
        result = run_aria(probe)
        if result["failure"] is not None:
            print(f"FAILED: {result['failure']['message']}")
        else:
            print(f"confidence: {result['confidence']}")
            print(result["answer"])
        print()
