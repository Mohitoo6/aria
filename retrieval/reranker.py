"""
Source-balanced retrieval with Cohere reranking.

The rerank model ID comes from `llm.config` rather than being hardcoded, so
the next reranker deprecation is an environment change, not a code change.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_cohere import CohereRerank
from pydantic import SecretStr
from qdrant_client.models import FieldCondition, Filter, MatchValue

from llm.config import rerank_model
from vectorstore.qdrant_store import load_vectorstore

logger = logging.getLogger(__name__)

__all__ = ["BalancedRetriever", "get_balanced_retriever"]

# Qdrant filter that matches only RxPrep chunks (payload field metadata.book)
RXPREP_FILTER = Filter(must=[FieldCondition(key="metadata.book", match=MatchValue(value="rxprep"))])


class BalancedRetriever:
    """
    Source-balanced retrieval across both books.

    The store holds far more DiPiro vectors than RxPrep, so a plain top-k
    search is numerically dominated by DiPiro and RxPrep never reaches the
    reranker. Here we pull a global candidate set AND a guaranteed RxPrep
    set (via metadata filter), merge them, and let Cohere rerank decide what
    is genuinely most relevant — so both books always get a fair hearing.
    """

    def __init__(
        self,
        vectorstore: Any,
        reranker: CohereRerank,
        k_global: int = 14,
        k_rxprep: int = 8,
    ) -> None:
        self.vs = vectorstore
        self.reranker = reranker
        self.k_global = k_global
        self.k_rxprep = k_rxprep

    def invoke(self, query: str) -> list[Any]:
        glob = self.vs.similarity_search(query, k=self.k_global)
        rx = self.vs.similarity_search(query, k=self.k_rxprep, filter=RXPREP_FILTER)

        seen: set[str] = set()
        candidates: list[Any] = []
        for d in glob + rx:
            key = d.page_content[:120]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(d)

        if not candidates:
            return []

        reranked = list(self.reranker.compress_documents(candidates, query))
        n_rx = sum(1 for d in reranked if d.metadata.get("book") == "rxprep")
        logger.info(
            "Balanced retrieve: %d candidates -> %d kept (%d RxPrep, %d DiPiro)",
            len(candidates),
            len(reranked),
            n_rx,
            len(reranked) - n_rx,
        )
        return reranked


def get_balanced_retriever(
    k_global: int = 14,
    k_rxprep: int = 8,
    top_n: int = 5,
) -> BalancedRetriever:
    # vectorstore/ is untyped and out of scope for this change.
    vectorstore: Any = load_vectorstore()  # type: ignore[no-untyped-call]
    model = rerank_model()
    api_key = os.getenv("COHERE_API_KEY")
    reranker = CohereRerank(
        model=model,
        top_n=top_n,
        cohere_api_key=SecretStr(api_key) if api_key else None,
    )
    logger.info(
        "Balanced retriever ready — global %d + RxPrep %d, rerank to top %d via %s",
        k_global,
        k_rxprep,
        top_n,
        model,
    )
    return BalancedRetriever(vectorstore, reranker, k_global, k_rxprep)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    retriever = get_balanced_retriever()
    demo_query = "what is diabetes mellitus"

    print(f"\nQuery: {demo_query}")
    results = retriever.invoke(demo_query)

    print(f"\n-------top {len(results)} reranked chunks------")
    for i, doc in enumerate(results):
        score = doc.metadata.get("relevance_score", "N/A")
        print(f"\nRank {i + 1} (page {doc.metadata.get('page', '?')}, score: {score}):")
        print(f"{doc.page_content[:1000]}")
