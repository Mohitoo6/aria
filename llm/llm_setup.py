"""
Groq client construction and invocation.
------------------------------------------------------------------
`get_llm` builds a client for a *role*; `invoke_role` runs a prompt through
it and guarantees one of two outcomes — clean text, or an `AriaLLMError`.
It never returns an error string as if it were model output.

Fallback (step 6): if a role's primary model reports `model_not_found` or
`model_decommissioned`, the call is retried once against
`ARIA_FALLBACK_MODEL` and a warning is logged. That is what turns the next
provider deprecation from an outage into a logged degradation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from llm.config import ModelSpec, Role, fallback_model, spec_for
from llm.errors import AriaLLMError, wrap_provider_error

load_dotenv()

logger = logging.getLogger(__name__)

__all__ = ["build_llm", "get_llm", "invoke_role"]


def build_llm(spec: ModelSpec) -> ChatGroq:
    """Construct a Groq chat client from an explicit spec."""
    return ChatGroq(
        model=spec.model,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        reasoning_effort=spec.reasoning_effort,
        api_key=os.getenv("GROQ_API_KEY"),  # type: ignore[arg-type]
    )


def get_llm(role: Role) -> ChatGroq:
    """Client for a pipeline role, configured from `llm.config`.

    The signature takes a role rather than a bare temperature so that no
    call site can pick a model — that decision lives in one registry.
    """
    return build_llm(spec_for(role))


def _as_text(content: str | list[str | dict[str, Any]]) -> str:
    """Flatten a LangChain message payload to plain text.

    Groq returns a plain string, but the base type also allows a list of
    content blocks; handle both so a provider change cannot surprise us.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _invoke_once(llm: ChatGroq, prompt: str) -> str:
    message: BaseMessage = llm.invoke(prompt)
    return _as_text(message.content)


def invoke_role(role: Role, prompt: str) -> str:
    """Run `prompt` for `role` and return the model's text.

    Raises:
        AriaLLMError: on any provider failure, including after the fallback
            model has also failed. Callers must let this propagate rather
            than substituting the message into an answer.
    """
    spec = spec_for(role)
    try:
        return _invoke_once(build_llm(spec), prompt)
    except Exception as exc:  # noqa: BLE001 - normalised immediately below
        primary_error = wrap_provider_error(exc, role.value, spec.model)

    secondary = fallback_model()
    if not primary_error.is_dead_model or secondary == spec.model:
        logger.error(
            "%s call failed on %s (code=%s)",
            role.value,
            spec.model,
            primary_error.code,
        )
        raise primary_error

    logger.warning(
        "Model %r is unavailable (code=%s) for role %r — falling back to %r. "
        "Set %s to a live model ID to silence this.",
        spec.model,
        primary_error.code,
        role.value,
        secondary,
        spec.env_var,
    )
    fallback_spec = ModelSpec(
        role=spec.role,
        model=secondary,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        reasoning_effort=spec.reasoning_effort,
        env_var=spec.env_var,
    )
    try:
        return _invoke_once(build_llm(fallback_spec), prompt)
    except Exception as exc:  # noqa: BLE001 - normalised immediately below
        fallback_error = wrap_provider_error(exc, role.value, secondary)
        logger.error(
            "Fallback model %r also failed for role %r (code=%s)",
            secondary,
            role.value,
            fallback_error.code,
        )
        raise fallback_error from primary_error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for probe_role in Role:
        probe_spec = spec_for(probe_role)
        try:
            reply = invoke_role(probe_role, "Reply with the single word OK.")
        except AriaLLMError as failure:
            print(f"{probe_spec.describe():<70} FAILED code={failure.code}")
        else:
            print(f"{probe_spec.describe():<70} -> {reply.strip()[:40]!r}")
