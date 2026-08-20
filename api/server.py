"""
ARIA bridge server.
------------------------------------------------------------------
Runs the real pipeline (guardrail -> navigator -> generator -> judge) and
streams it to the web frontend as Server-Sent Events whose shape matches
`web/src/lib/client.ts` (ConsultationEvent).

Event contract, and the reason it looks like this:

  steps  — agent trace; a step may end `done`, `skipped` or `failed`
  meta   — evidence tier, confidence, citations. Emitted ONLY when a real
           grounded answer exists. `confidence: null` means the answer is
           real but was not adjudicated.
  token  — a fragment of grounded answer prose, and nothing else. An
           exception message must never travel on this channel: the UI
           renders tokens as the assistant's reply and decorates them with
           an evidence tier, so an error sent as a token is presented to a
           clinician with the full authority of a cited answer.
  error  — a failure. Terminal, carries no confidence and no citations.
  done   — end of turn.

Run from the project root:
    aria_env/bin/uvicorn api.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

# Allow running as `python api/server.py` as well as `uvicorn api.server:app`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.guardrail_agent import check_guardrail
from agents.judge_agent import judge_answer
from agents.navigator_agent import navigator
from llm.errors import AriaLLMError, wrap_provider_error
from llm.generator import generate_answer
from llm.preflight import PreflightReport, run_preflight

logging.basicConfig(
    level=os.getenv("ARIA_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("aria.api")

#: Result of the boot-time model check, exposed on /api/health.
_preflight: PreflightReport = PreflightReport()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate every configured model before serving a single request."""
    global _preflight
    _preflight = await asyncio.to_thread(run_preflight)
    yield


app = FastAPI(title="ARIA Bridge", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConsultRequest(BaseModel):
    query: str


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def base_steps() -> list[dict[str, Any]]:
    return [
        {
            "id": "guardrail",
            "label": "Guardrail",
            "detail": "Confirming the query is in clinical scope",
            "status": "pending",
        },
        {
            "id": "navigator",
            "label": "Navigator",
            "detail": "Retrieving & reranking DiPiro passages",
            "status": "pending",
        },
        {
            "id": "generator",
            "label": "Generator",
            "detail": "Synthesizing a grounded answer",
            "status": "pending",
        },
        {
            "id": "judge",
            "label": "Judge",
            "detail": "Scoring faithfulness & evidence strength",
            "status": "pending",
        },
    ]


def tier_from_confidence(c: float) -> str:
    if c >= 0.8:
        return "strong"
    if c >= 0.6:
        return "moderate"
    return "limited"


def tier_from_relevance(r: Any) -> str:
    try:
        value = float(r)
    except (TypeError, ValueError):
        return "moderate"
    if value >= 0.75:
        return "strong"
    if value >= 0.5:
        return "moderate"
    return "limited"


# How each source book is cited and labelled in the UI.
BOOKS: dict[str, dict[str, str]] = {
    "dipiro": {
        "source": "DiPiro's Pharmacotherapy",
        "section": "A Pathophysiologic Approach, 12e",
    },
    "rxprep": {
        "source": "RxPrep NAPLEX Course Book (2025)",
        "section": "UWorld · NAPLEX review",
    },
}


def build_citations(chunks: list[Any]) -> list[dict[str, Any]]:
    cites: list[dict[str, Any]] = []
    for i, doc in enumerate(chunks):
        meta = getattr(doc, "metadata", {}) or {}
        rel = meta.get("relevance_score", meta.get("score"))
        snippet = re.sub(r"\s+", " ", (getattr(doc, "page_content", "") or "")).strip()
        if len(snippet) > 360:
            snippet = snippet[:357].rstrip() + "…"
        page = meta.get("page", "?")
        # Provenance: chunks ingested with a `book` tag (RxPrep); DiPiro
        # predates the tag, so default to it when absent.
        book = meta.get("book", "dipiro")
        info = BOOKS.get(book, BOOKS["dipiro"])
        cites.append(
            {
                "id": f"c{i + 1}",
                "marker": i + 1,
                "book": book,
                # Always prefer the curated label over raw file-path metadata.
                "source": info["source"],
                "section": meta.get("section") or info["section"],
                "page": f"p. {page}" if page != "?" else "—",
                "snippet": snippet or "(no text)",
                "relevance": round(float(rel), 2) if rel is not None else 0.0,
                "tier": tier_from_relevance(rel),
            }
        )
    return cites


def error_event(exc: AriaLLMError) -> dict[str, Any]:
    """The one and only way a failure reaches the browser.

    Note what is absent: no confidence, no evidenceTier, no citations, and
    nothing that the UI could render as answer prose.
    """
    return {
        "type": "error",
        "stage": exc.stage,
        "code": exc.code or "provider_error",
        "message": exc.public_message(),
    }


async def stream_tokens(text: str) -> AsyncIterator[str]:
    for tok in re.findall(r"\s+|\S+", text):
        yield sse({"type": "token", "chunk": tok})
        await asyncio.sleep(0.014 if not re.match(r"[.,;:]", tok) else 0.03)


async def run_consultation(query: str) -> AsyncIterator[str]:
    steps = base_steps()

    def patch(step_id: str, **kw: Any) -> None:
        for s in steps:
            if s["id"] == step_id:
                s.update(kw)

    def steps_event() -> str:
        return sse({"type": "steps", "steps": [dict(s) for s in steps]})

    def fail_from(step_id: str, exc: AriaLLMError) -> list[str]:
        """Mark the failing step and every step after it, then report."""
        patch(step_id, status="failed", detail="Failed — no answer produced", metric=None)
        seen_failing = False
        for s in steps:
            if s["id"] == step_id:
                seen_failing = True
                continue
            if seen_failing and s["status"] == "pending":
                s.update(status="skipped", detail="Not reached")
        logger.error("consultation failed at %s: %s", step_id, exc)
        return [steps_event(), sse(error_event(exc)), sse({"type": "done"})]

    # 1 — Guardrail
    patch("guardrail", status="active")
    yield steps_event()
    t0 = time.time()
    try:
        is_medical = await asyncio.to_thread(check_guardrail, query)
    except AriaLLMError as exc:
        # Previously this defaulted to `is_medical = True`, so an unreachable
        # guardrail silently disabled the clinical scope filter.
        for ev in fail_from("guardrail", exc):
            yield ev
        return
    patch(
        "guardrail",
        status="done",
        durationMs=int((time.time() - t0) * 1000),
        metric="medical · in scope" if is_medical else "out of scope",
        detail=(
            "Clinical pharmacotherapy query" if is_medical else "Query is outside clinical scope"
        ),
    )
    yield steps_event()

    if not is_medical:
        for sid in ("navigator", "generator", "judge"):
            patch(sid, status="skipped", detail="Skipped — out of scope")
        yield steps_event()
        yield sse(
            {
                "type": "meta",
                "evidenceTier": "limited",
                "confidence": 0,
                "citations": [],
                "safety": [
                    {
                        "kind": "scope",
                        "text": (
                            "ARIA answers pharmacotherapy questions only, grounded "
                            "in DiPiro's Pharmacotherapy."
                        ),
                    }
                ],
            }
        )
        msg = (
            "That falls outside my scope. I'm **ARIA**, a clinical pharmacotherapy "
            "assistant — I can help with drug selection, dosing, monitoring, "
            "interactions, and the evidence behind therapeutic decisions, grounded "
            "in *DiPiro's Pharmacotherapy*."
        )
        async for ev in stream_tokens(msg):
            yield ev
        yield sse({"type": "done"})
        return

    # 2 — Navigator (query rewrite + source-balanced retrieve + Cohere rerank)
    patch("navigator", status="active")
    yield steps_event()
    t0 = time.time()
    try:
        chunks = await asyncio.to_thread(navigator, query)
    except Exception as exc:  # noqa: BLE001 - normalised into an error event
        for ev in fail_from("navigator", wrap_provider_error(exc, "navigator", "retrieval")):
            yield ev
        return
    patch(
        "navigator",
        status="done",
        durationMs=int((time.time() - t0) * 1000),
        metric=f"retrieved → {len(chunks)} reranked",
        detail="Top passages selected by relevance",
    )
    yield steps_event()

    # 3 — Generator
    patch("generator", status="active")
    yield steps_event()
    t0 = time.time()
    try:
        answer = await asyncio.to_thread(generate_answer, query, chunks)
    except AriaLLMError as exc:
        for ev in fail_from("generator", exc):
            yield ev
        return
    patch(
        "generator",
        status="done",
        durationMs=int((time.time() - t0) * 1000),
        metric=f"{len(chunks)} sources cited",
        detail="Answer grounded in retrieved passages",
    )
    yield steps_event()

    # 4 — Judge (computed before reveal so tier/confidence are real)
    patch("judge", status="active")
    yield steps_event()
    t0 = time.time()
    judge_error: AriaLLMError | None = None
    confidence: float | None = None
    try:
        judgment = await asyncio.to_thread(judge_answer, query, answer, chunks)
        confidence = judgment.confidence
    except AriaLLMError as exc:
        # The answer and its citations are genuine — only the score is
        # missing. Never substitute a placeholder number here; a fabricated
        # 0.5 used to be drawn on a real calibrated gauge.
        judge_error = exc
        logger.warning("judge unavailable (%s) — answer left unadjudicated", exc.code)

    adjudicated = confidence is not None
    yield sse(
        {
            "type": "meta",
            "evidenceTier": tier_from_confidence(confidence) if confidence is not None else None,
            "confidence": confidence,
            "citations": build_citations(chunks),
            "safety": [
                {
                    "kind": "caution",
                    "text": (
                        "Generated from textbook evidence — verify against current "
                        "guidelines and patient context."
                    ),
                }
            ],
        }
    )

    async for ev in stream_tokens(answer):
        yield ev

    if adjudicated and confidence is not None:
        patch(
            "judge",
            status="done",
            durationMs=int((time.time() - t0) * 1000),
            metric=f"{round(confidence * 100)}% confidence",
            detail="Answer scored for faithfulness to cited sources",
        )
    else:
        patch(
            "judge",
            status="failed",
            durationMs=int((time.time() - t0) * 1000),
            metric="not adjudicated",
            detail=(
                "Judge unavailable — answer not scored"
                if judge_error is not None
                else "Judge returned no usable score"
            ),
        )
    yield steps_event()
    yield sse({"type": "done"})


@app.get("/api/health")
async def health() -> JSONResponse:
    """Health, including whether every configured model exists upstream."""
    return JSONResponse(
        {
            "status": "ok" if _preflight.ok else "degraded",
            "backend": "aria-langgraph",
            "models": _preflight.as_dict(),
        }
    )


@app.post("/api/consult")
async def consult(req: ConsultRequest) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        try:
            async for ev in run_consultation(req.query.strip()):
                yield ev
        except Exception as exc:  # last resort — still never a token
            # The old code streamed `str(exc)` as a `token`, so the browser
            # rendered a stack trace as ARIA's grounded reply. Failures leave
            # through the error channel or not at all.
            logger.exception("unhandled consultation error")
            yield sse(error_event(wrap_provider_error(exc, "consultation", "unknown")))
            yield sse({"type": "done"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the built web frontend (web/dist) when it exists, so a single
# process can host both the API and the UI in production. Registered after
# the /api routes, so those keep precedence.
_DIST = os.path.join(ROOT, "web", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
