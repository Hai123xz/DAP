"""
Step 3 — MASIVE Fine-Grained Labeling (API-Based)

Converts verified RVISA reasoning into fine-grained affective state labels:
  - fine_grained_label (short phrase, 2-6 words)
  - alt_labels (top-k alternatives)
  - confidence
  - normalization (exact match / embedding / clustering)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from call_llm_mistral import ask

from modules.models import (
    RVISAOutput,
    MASIVEOutput,
    NormalizationInfo,
    NormalizationMethod,
    Verdict,
)

logger = logging.getLogger(__name__)


# ========================= PROMPT =========================

MASIVE_SYSTEM_PROMPT = """You are a MASIVE-style open-ended affective state identifier. Your task is to generate the most precise, psychologically meaningful affective-state label for a given verified rationale.

Label constraints:
- 2–6 words (English) or 2–8 words (Vietnamese)
- Avoid overly generic labels (e.g., "negative", "sad")
- Prefer psychologically meaningful phrases, for example:
  "betrayed", "dismissive resentment", "frustrated due to neglect", "anxious uncertainty",
  "grateful relief", "bitter disappointment", "defensive hostility"

Also provide 2 alternative labels and a confidence score.

Return ONLY valid JSON. No extra text, no markdown fences.

Schema:
{
  "fine_grained_label": "string — the most precise affective state label",
  "alt_labels": ["string", "string"],
  "confidence": 0.0
}"""

MASIVE_USER_TEMPLATE = """Given the verified rationale below, output the most precise affective-state label as a short phrase.

Verified rationale: "{verified_rationale}"
Aspect: "{aspect}"
Cause: "{cause}"
Attitude: "{attitude}"
"""


# ========================= JSON PARSING =========================

def _try_parse_json(text: str) -> Optional[dict]:
    """Try to parse JSON from model output."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        lines = [l for l in lines if not l.strip().startswith('```')]
        text = '\n'.join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ========================= MAIN FUNCTION =========================

def run_masive(
    rvisa_output: RVISAOutput,
    model_name: str = "mistral-small-latest",
    temperature: float = 0.3,
    max_retries: int = 2,
) -> MASIVEOutput:
    """
    Step 3: Generate fine-grained affective state label from verified reasoning.
    
    Only runs on RVISA PASS results. For FAIL results, returns empty label.
    
    Args:
        rvisa_output: Verified output from Step 2.
        model_name: Mistral model to use.
        temperature: Higher temp for more creative labeling.
        max_retries: Retries on JSON parse failure.
        
    Returns:
        MASIVEOutput with fine-grained label and normalization.
    """
    conv_id = rvisa_output.conversation_id
    utt_id = rvisa_output.utt_id

    # Skip if RVISA failed
    if rvisa_output.verdict == Verdict.FAIL:
        logger.info("Skipping MASIVE for anchor %s (RVISA verdict: FAIL)", utt_id)
        return MASIVEOutput(
            conversation_id=conv_id,
            utt_id=utt_id,
            fine_grained_label="unknown",
            confidence=0.0,
        )

    user_prompt = MASIVE_USER_TEMPLATE.format(
        verified_rationale=rvisa_output.verified_rationale,
        aspect=rvisa_output.aspect,
        cause=rvisa_output.cause,
        attitude=rvisa_output.inferred_attitude,
    )

    parsed = None
    for attempt in range(max_retries + 1):
        try:
            response = ask(
                user_prompt=user_prompt,
                sys_prompt=MASIVE_SYSTEM_PROMPT,
                model_name=model_name,
                temperature=temperature,
                infinite_retry=True,
            )

            parsed = _try_parse_json(response)
            if parsed is not None:
                break

            logger.warning(
                "Step 3 attempt %d: JSON parse failed for anchor %s",
                attempt + 1, utt_id,
            )
        except Exception as e:
            logger.error("Step 3 attempt %d failed: %s", attempt + 1, e)

    if parsed is None:
        logger.error("Step 3: All retries exhausted for anchor %s", utt_id)
        return MASIVEOutput(
            conversation_id=conv_id,
            utt_id=utt_id,
            fine_grained_label="unknown",
            confidence=0.0,
        )

    label = parsed.get('fine_grained_label', 'unknown')
    alt_labels = parsed.get('alt_labels', [])
    confidence = float(parsed.get('confidence', 0.0))

    # Normalization: exact match (simple approach)
    normalization = NormalizationInfo(
        method=NormalizationMethod.EXACT,
        matched_vocab_id=None,
        cluster_id=None,
    )

    return MASIVEOutput(
        conversation_id=conv_id,
        utt_id=utt_id,
        fine_grained_label=label,
        alt_labels=alt_labels if isinstance(alt_labels, list) else [],
        confidence=confidence,
        normalization=normalization,
    )
