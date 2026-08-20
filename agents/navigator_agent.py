"""
Navigator agent — rewrites the query, then retrieves and reranks passages.

The query rewrite is an optimisation, not a source of clinical content: if
the rewrite model is unavailable the original question is used instead and
a warning is logged. Retrieval still runs against the real corpus, so every
passage the generator sees remains genuine. Retrieval failures themselves
are not caught here — there is no safe way to answer without sources.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llm.config import Role
from llm.errors import AriaLLMError
from llm.llm_setup import invoke_role
from retrieval.reranker import get_balanced_retriever

if TYPE_CHECKING:
    from retrieval.reranker import BalancedRetriever

logger = logging.getLogger(__name__)

__all__ = ["navigator", "optimize_query"]

OPTIMIZE_PROMPT = """You are a medical search query optimizer.
Convert the user's question into a clear, concise medical search query.
Remove personal details, keep only the core medical concept.

Respond with ONLY the optimized query, nothing else.

User question: {query}

Optimized query:"""

# Built once and reused: creating it loads the embedding model and opens
# the Qdrant connection, which is far too slow to repeat per query.
_retriever: BalancedRetriever | None = None


def optimize_query(query: str) -> str:
    """Rewrite `query` as a focused search string.

    Degrades to the original question if the rewrite model is unavailable —
    a worse search, but still a real one over real sources.
    """
    try:
        optimized = invoke_role(Role.NAVIGATOR, OPTIMIZE_PROMPT.format(query=query)).strip()
    except AriaLLMError as exc:
        logger.warning(
            "Query optimisation unavailable (%s) — retrieving with the raw question",
            exc.code or "provider error",
        )
        return query

    if not optimized:
        logger.warning("Query optimiser returned nothing — using the raw question")
        return query

    logger.info("original : %s", query)
    logger.info("optimized : %s", optimized)
    return optimized


def navigator(query: str) -> list[Any]:
    """Retrieve the reranked passages that should ground the answer."""
    global _retriever
    if _retriever is None:
        _retriever = get_balanced_retriever()

    chunks = _retriever.invoke(optimize_query(query))
    logger.info("%d relevant chunks retrieved", len(chunks))
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    user_query = "My grandmother has high blood pressure, what treatment is there?"

    print("-------Navigator testing------\n")
    retrieved = navigator(user_query)

    print("--------Retrieved chunks--------\n")
    for i, doc in enumerate(retrieved):
        print(f"\nchunk {i + 1} (page {doc.metadata.get('page', '?')}):")
        print(f"{doc.page_content[:500]}")
