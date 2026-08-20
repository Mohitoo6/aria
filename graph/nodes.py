"""
LangGraph node functions.

Every node that touches an LLM converts a provider failure into an explicit
`failure` entry on the state and stops the pipeline. No node ever writes an
error message into `answer` — that field is reserved for grounded content,
and the graph's routing depends on being able to tell the two apart.
"""

from __future__ import annotations

import logging

from agents.guardrail_agent import check_guardrail
from agents.judge_agent import judge_answer
from agents.navigator_agent import navigator
from graph.state import AriaState, record_failure
from llm.errors import AriaLLMError
from llm.generator import generate_answer

logger = logging.getLogger(__name__)

__all__ = [
    "generator_node",
    "guardrail_node",
    "judge_node",
    "navigator_node",
    "reject_node",
]

REJECTION_MESSAGE = (
    "I'm ARIA, a medical assistant. I can only help with medical and pharmacology questions."
)


def guardrail_node(state: AriaState) -> AriaState:
    logger.info("GUARDRAIL")
    try:
        state["is_medical"] = check_guardrail(state["query"])
    except AriaLLMError as exc:
        # Deliberately NOT defaulting to in-scope: an unreachable guardrail
        # cannot vouch for the query, so the consultation stops here.
        record_failure(state, exc)
    return state


def navigator_node(state: AriaState) -> AriaState:
    logger.info("NAVIGATOR")
    try:
        state["chunks"] = navigator(state["query"])
    except AriaLLMError as exc:
        record_failure(state, exc)
    return state


def generator_node(state: AriaState) -> AriaState:
    logger.info("GENERATOR")
    try:
        state["answer"] = generate_answer(state["query"], state["chunks"])
    except AriaLLMError as exc:
        record_failure(state, exc)
    return state


def judge_node(state: AriaState) -> AriaState:
    logger.info("JUDGE")
    state["retry_count"] = state["retry_count"] + 1
    try:
        judgment = judge_answer(state["query"], state["answer"], state["chunks"])
    except AriaLLMError as exc:
        # The answer itself is real and grounded; only adjudication failed.
        # Leave `answer` intact, mark it unadjudicated, and let the caller
        # present it without a fabricated score.
        logger.warning("judge unavailable (%s) — answer left unadjudicated", exc.code)
        state["confidence"] = None
        state["judge_failed"] = True
        return state

    state["confidence"] = judgment.confidence
    state["judge_failed"] = False
    return state


def reject_node(state: AriaState) -> AriaState:
    logger.info("REJECT")
    state["answer"] = REJECTION_MESSAGE
    return state
