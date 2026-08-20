"""
Generator — synthesises the answer from retrieved passages only.

Returns grounded prose or raises. It must never return an error string:
the whole point of the error taxonomy is that the caller can tell the
difference between an answer and a failure without inspecting the text.
"""

from __future__ import annotations

import logging
from typing import Any

from llm.config import Role
from llm.llm_setup import invoke_role
from llm.prompts import ANSWER_PROMPT

logger = logging.getLogger(__name__)

__all__ = ["generate_answer"]


def generate_answer(query: str, chunks: list[Any]) -> str:
    """Write an answer to `query` grounded strictly in `chunks`.

    Raises:
        AriaLLMError: if the generator model (and its fallback) cannot be
            reached. The exception text is never a valid answer.
    """
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    answer = invoke_role(
        Role.GENERATOR,
        ANSWER_PROMPT.format(context=context, question=query),
    )
    logger.info("generated answer of %d characters from %d chunks", len(answer), len(chunks))
    return answer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from agents.navigator_agent import navigator

    demo_query = (
        "my mother is 50 years old and she is having diabetes since last seven years. "
        "She is on oral hypoglycaemic drugs like glycomet gp 1 twice daily and "
        "vildagliptin 50 MG twice daily, but in the morning blood sugar level often "
        "remains increase what we could do to control that?"
    )

    print("Getting chunks from navigator\n")
    demo_chunks = navigator(demo_query)

    print("\n Generating Answer...")
    print("\n=========Answer============")
    print(generate_answer(demo_query, demo_chunks))
