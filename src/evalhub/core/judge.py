"""LLM-as-judge scoring.

The platform can score each sample with a "judge" that produces a 0-1 quality
score plus a short rationale. Two backends are provided:

* ``HeuristicJudge`` — deterministic and fully offline. It combines token
  overlap with a length signal. This is the default so the whole system runs
  and tests without any network access or API key.
* ``AnthropicJudge`` — calls Claude (``claude-opus-4-8`` by default) with a
  structured-output schema so the response is guaranteed to be valid JSON.
  Used only when explicitly configured *and* an API key is available.

Both implement the async ``Judge`` protocol, so the evaluation engine treats
them identically.
"""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..config import Settings
from .models import JudgeBackend, JudgeConfig, Sample
from .scorers import token_f1, tokenize


class JudgeResult(BaseModel):
    """A judge's verdict for one sample."""

    score: float  # in [0, 1]
    reasoning: str


@runtime_checkable
class Judge(Protocol):
    """Anything that can asynchronously score a single sample."""

    async def score(self, sample: Sample) -> JudgeResult: ...


class HeuristicJudge:
    """Offline judge: token-F1 blended with a length-sanity signal.

    Deterministic, dependency-free, and fast — ideal as a default and for
    tests. It approximates "did the prediction cover the reference without
    rambling" without any model call.
    """

    def __init__(self, config: JudgeConfig) -> None:
        self._config = config

    async def score(self, sample: Sample) -> JudgeResult:
        if sample.reference is None:
            # With no reference, fall back to a pure "is there content" signal.
            score = 1.0 if sample.prediction.strip() else 0.0
            return JudgeResult(
                score=score,
                reasoning="No reference provided; scored on non-emptiness only.",
            )

        overlap = token_f1(sample.prediction, sample.reference)
        pred_len = len(tokenize(sample.prediction))
        ref_len = max(len(tokenize(sample.reference)), 1)
        # Penalize answers that are wildly longer than the reference.
        verbosity_penalty = min(pred_len / ref_len, 3.0)
        length_signal = 1.0 if verbosity_penalty <= 1.5 else 1.5 / verbosity_penalty
        score = round(0.8 * overlap + 0.2 * length_signal, 4)
        return JudgeResult(
            score=score,
            reasoning=(
                f"token_f1={overlap:.2f}, length_signal={length_signal:.2f} "
                f"(pred={pred_len} tok, ref={ref_len} tok)."
            ),
        )


# JSON schema handed to Claude to constrain the judge's structured output.
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "Quality from 0.0 (poor) to 1.0 (excellent).",
        },
        "reasoning": {
            "type": "string",
            "description": "One or two sentences justifying the score.",
        },
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


class AnthropicJudge:
    """LLM judge backed by the Anthropic API.

    Uses adaptive thinking and structured outputs per the current API surface.
    The ``anthropic`` package is imported lazily so the platform runs without
    it installed when only the heuristic backend is used.
    """

    def __init__(self, config: JudgeConfig, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "AnthropicJudge requires EVALHUB_ANTHROPIC_API_KEY to be set."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "The 'anthropic' package is not installed. Install the "
                "'llm' extra: pip install 'evalhub[llm]'."
            ) from exc

        from anthropic import AsyncAnthropic

        self._config = config
        self._model = settings.anthropic_model
        self._timeout = settings.judge_timeout_seconds
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def score(self, sample: Sample) -> JudgeResult:
        prompt = self._build_prompt(sample)
        response = await self._client.with_options(
            timeout=self._timeout
        ).messages.create(
            model=self._model,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            output_config={
                "format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}
            },
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), None
        )
        if text is None:  # pragma: no cover - defensive
            raise RuntimeError("judge response contained no text block")
        data = json.loads(text)
        # Clamp to the valid range in case the model drifts.
        score = max(0.0, min(1.0, float(data["score"])))
        return JudgeResult(score=score, reasoning=str(data["reasoning"]))

    def _build_prompt(self, sample: Sample) -> str:
        parts = [
            "You are an impartial evaluator. Score the response strictly.",
            f"Rubric: {self._config.criteria}",
            "",
        ]
        if sample.input:
            parts.append(f"Input:\n{sample.input}\n")
        if sample.reference:
            parts.append(f"Reference answer:\n{sample.reference}\n")
        parts.append(f"Response to evaluate:\n{sample.prediction}")
        return "\n".join(parts)


def build_judge(config: JudgeConfig, settings: Settings) -> Judge:
    """Construct a judge, falling back to heuristic if Anthropic is unavailable.

    A configured Anthropic backend without an API key degrades gracefully to
    the heuristic judge rather than failing the whole evaluation.
    """
    if config.backend is JudgeBackend.anthropic and settings.anthropic_api_key:
        return AnthropicJudge(config, settings)
    return HeuristicJudge(config)
