"""
ARIA failure taxonomy.
------------------------------------------------------------------
The rule this module exists to enforce: a provider failure is never a
clinical answer.

Before this, an exception's ``str()`` was streamed to the browser through
the same channel as generated prose, so the UI stamped it with an evidence
tier and a "grounded reply" byline. For a clinical decision-support tool
that is not a cosmetic bug — it presents an error string with the visual
authority of adjudicated, cited medical guidance.

So failures travel as their own type, all the way to their own SSE event
and their own UI state. Nothing here ever produces text that could be
mistaken for an answer.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEAD_MODEL_CODES",
    "AriaError",
    "AriaLLMError",
    "AriaPreflightError",
    "error_code_of",
    "wrap_provider_error",
]

#: Provider error codes meaning "this model ID is gone" — the exact
#: condition that took ARIA down when Groq retired llama-3.3-70b-versatile.
#: These are the codes that trigger the fallback model.
DEAD_MODEL_CODES: frozenset[str] = frozenset({"model_not_found", "model_decommissioned"})


class AriaError(Exception):
    """Base class for every failure ARIA raises deliberately."""


class AriaPreflightError(AriaError):
    """Raised when startup validation finds a configured model missing."""


class AriaLLMError(AriaError):
    """An LLM/provider call failed.

    Carries enough structure for the API layer to build an honest error
    event: which pipeline stage broke, which model, and the provider's own
    error code. It deliberately does NOT carry anything answer-shaped.
    """

    def __init__(
        self,
        stage: str,
        model: str,
        message: str,
        code: str | None = None,
    ) -> None:
        self.stage = stage
        self.model = model
        self.provider_message = message
        self.code = code
        super().__init__(f"[{stage}] {model}: {message}")

    @property
    def is_dead_model(self) -> bool:
        """True when the model ID itself no longer exists at the provider."""
        return self.code in DEAD_MODEL_CODES

    def public_message(self) -> str:
        """A reader-facing sentence. Never mistakable for clinical content.

        Kept free of stack traces and provider jargon: it states that no
        answer was produced and why, and stops there.
        """
        if self.is_dead_model:
            return (
                f"ARIA could not produce an answer: the {self.stage} model "
                f"({self.model}) is no longer available from the provider. "
                "No clinical content was generated."
            )
        return (
            f"ARIA could not produce an answer: the {self.stage} step failed "
            "to reach the language model. No clinical content was generated."
        )


def error_code_of(exc: BaseException) -> str | None:
    """Best-effort extraction of a provider error code.

    Groq's SDK raises ``groq.NotFoundError`` carrying a parsed
    ``body = {"error": {"code": "model_not_found", ...}}``. Other providers
    and transports differ, so fall back to scanning the message for a known
    code rather than assuming a shape.
    """
    body: Any = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, str) and code:
                return code

    text = str(exc)
    for known in DEAD_MODEL_CODES:
        if known in text:
            return known
    return None


def wrap_provider_error(exc: BaseException, stage: str, model: str) -> AriaLLMError:
    """Normalise any provider exception into an :class:`AriaLLMError`."""
    if isinstance(exc, AriaLLMError):
        return exc
    return AriaLLMError(
        stage=stage,
        model=model,
        message=str(exc),
        code=error_code_of(exc),
    )
