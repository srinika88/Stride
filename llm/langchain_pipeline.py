"""
StrideIQ — LangChain Coaching Pipeline

Takes structured model output:
    - predicted label
    - confidence
    - biomechanical details

and converts it into personalized running coaching.

LangChain handles orchestration.
Ollama handles local LLM inference.

The rest of StrideIQ only calls:
    generate_coaching()
"""

import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from backend import config
from llm.prompts import (
    COACHING_SYSTEM_PROMPT,
    COACHING_USER_TEMPLATE
)
from ml.recommendation_engine import recommend as rule_based_recommend

logger = logging.getLogger("strideiq.langchain")


# ============================================================
# Build LangChain pipeline — built ONCE (lazily, on first use), not on
# every generate_coaching() call.
#
# Model: llama3.2:1b, not the default 3B llama3.2 — chosen specifically
# because this hasn't been latency-tested on the actual Raspberry Pi yet
# (only ~5-7s tested on a Mac terminal), and the Pi will be meaningfully
# slower. 1b trades some coaching-text nuance for real speed margin
# ahead of a live demo. Override via the LLM_MODEL env var if quality
# testing on the Pi says otherwise.
# ============================================================

def _build_chain():
    llm = OllamaLLM(
        model=config.LLM_MODEL or "llama3.2:1b",
        temperature=config.LLM_TEMPERATURE,
        num_predict=config.LLM_MAX_TOKENS,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )

    prompt = PromptTemplate(
        input_variables=[
            "related_metric", "predicted_label", "confidence",
            "details", "runner_history",
        ],
        template=COACHING_SYSTEM_PROMPT + "\n\n" + COACHING_USER_TEMPLATE,
    )

    return prompt | llm


_chain = None


def _get_chain():
    """Lazy singleton — built on first use, reused after that."""
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain


# ============================================================
# Public function
# ============================================================

def generate_coaching(
    related_metric: str,
    predicted_label: str,
    confidence: float,
    details: str,
    runner_history: str = "No prior context available.",
) -> str:
    """
    Structured biomechanics in. Personalized coaching text out.
    """
    if predicted_label == "Normal":
        return (
            "Your running form looks stable in this window. "
            "Keep maintaining your current rhythm."
        )

    try:
        chain = _get_chain()

        result = chain.invoke({
            "related_metric": related_metric or "general form",
            "predicted_label": predicted_label,
            "confidence": confidence,
            "details": details,
            "runner_history": runner_history,
        })

        return result.strip()

    except Exception as e:
        logger.exception(f"LangChain coaching failed: {e}")

        # Previously returned a hardcoded "unavailable" string here,
        # which meant backend/mqtt/feedback_loop.py's fallback to
        # ml.recommendation_engine.recommend() — added specifically to
        # cover this case — could never actually run, since this
        # function never let the exception escape for that fallback to
        # catch. Falling back HERE, at the source, means every caller
        # gets real usable coaching text instead of a dead-end message,
        # regardless of whether they wrap this call themselves.
        return rule_based_recommend(predicted_label)


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":
    test = generate_coaching(
        related_metric="impact",
        predicted_label="High Impact",
        confidence=0.82,
        details=(
            "Left foot impact 4.3g, right foot impact 2.8g. "
            "Runner is loading left side more."
        ),
    )
    print(test) 