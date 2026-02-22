"""
Step 2 — RVISA Reasoning Verification (API-Based)

Two-stage pipeline for each anchor utterance:
  2A. Generator: proposes structured reasoning (aspect → cause → attitude)
  2B. Verifier: checks if reasoning is supported by text evidence
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from call_llm_mistral import ask

from modules.models import (
    PreprocessedConversation,
    PreprocessedUtterance,
    TransMistralOutput,
    RVISAOutput,
    EvidenceSpan,
    InferredAttitude,
    Verdict,
    AnchorEntry,
    FlipEvent,
)

logger = logging.getLogger(__name__)


# ========================= WINDOW BUILDING =========================

def build_rvisa_window(
    anchor_utt_id: str,
    preprocessed: PreprocessedConversation,
    flip_events: List[FlipEvent],
    k_before: int = 6,
    k_after: int = 2,
    k_before_flip: int = 12,
    k_after_flip: int = 4,
) -> List[PreprocessedUtterance]:
    """
    Build context window around an anchor utterance.
    Expands window near flip zones.
    """
    utts = preprocessed.utterances
    anchor_idx = None
    for i, utt in enumerate(utts):
        if utt.utt_id == anchor_utt_id:
            anchor_idx = i
            break

    if anchor_idx is None:
        logger.warning("Anchor %s not found in conversation", anchor_utt_id)
        return []

    # Check if anchor is near a flip → expand window
    is_near_flip = False
    flip_utt_ids = set()
    for flip in flip_events:
        flip_utt_ids.update([flip.from_utt_id, flip.to_utt_id, flip.trigger_utt_id])
    if anchor_utt_id in flip_utt_ids:
        is_near_flip = True

    kb = k_before_flip if is_near_flip else k_before
    ka = k_after_flip if is_near_flip else k_after

    start = max(0, anchor_idx - kb)
    end = min(len(utts), anchor_idx + ka + 1)

    return utts[start:end]


def serialize_window(window: List[PreprocessedUtterance]) -> str:
    """Serialize window utterances for prompt."""
    lines = []
    for utt in window:
        text = utt.text_translated or utt.text_clean
        lines.append(f'[{utt.utt_id}][{utt.speaker_id}] "{text}"')
    return '\n'.join(lines)


# ========================= GENERATOR =========================

GENERATOR_SYSTEM_PROMPT = """You are RVISA Generator. Analyze the given context window and anchor utterance to infer implicit sentiment through structured reasoning.

Think internally, but output ONLY valid JSON matching the schema below. No extra text, no markdown fences.

Schema:
{
  "aspect": "string — what is being evaluated (e.g., service quality, response time)",
  "cause": "string — why the user feels that way",
  "inferred_attitude": "positive|negative|neutral|mixed",
  "rationale": "string — step-by-step reasoning explaining the inference",
  "evidence": [
    {"utt_id": "string", "quote": "string — exact text from the window supporting the inference"}
  ]
}"""

GENERATOR_USER_TEMPLATE = """Given a context window and an anchor utterance, infer implicit sentiment using 3-step reasoning:
(1) Aspect identification — what is the user evaluating?
(2) Cause analysis — why does the user feel that way?
(3) Attitude inference — what is the underlying attitude?

Global context summary: "{context_summary}"

Window:
{window_text}

Anchor:
[{anchor_id}][{anchor_speaker}] "{anchor_text}"
"""


def run_rvisa_generator(
    anchor_utt: PreprocessedUtterance,
    window: List[PreprocessedUtterance],
    context_summary: str,
    model_name: str = "mistral-small-latest",
    temperature: float = 0.1,
) -> Optional[dict]:
    """
    Step 2A: Run RVISA Generator to propose reasoning hypothesis.
    Returns parsed JSON dict or None.
    """
    window_text = serialize_window(window)
    anchor_text = anchor_utt.text_translated or anchor_utt.text_clean

    user_prompt = GENERATOR_USER_TEMPLATE.format(
        context_summary=context_summary,
        window_text=window_text,
        anchor_id=anchor_utt.utt_id,
        anchor_speaker=anchor_utt.speaker_id,
        anchor_text=anchor_text,
    )

    try:
        response = ask(
            user_prompt=user_prompt,
            sys_prompt=GENERATOR_SYSTEM_PROMPT,
            model_name=model_name,
            temperature=temperature,
            infinite_retry=True,
        )
        return _try_parse_json(response)
    except Exception as e:
        logger.error("RVISA Generator failed: %s", e)
        return None


# ========================= VERIFIER =========================

VERIFIER_SYSTEM_PROMPT = """You are RVISA Verifier. Your job is to verify whether the generator's reasoning is actually supported by evidence in the conversation window.

PASS if ALL are true:
- aspect/cause are explicitly or strongly implied in the window
- rationale does not invent new events/entities
- evidence quotes point to real text content
- speaker references are consistent

FAIL if ANY are true:
- rationale relies on unsupported assumptions
- contradicts text in the window
- wrong speaker attribution
- sarcasm is "interpreted" without contextual evidence

Output ONLY valid JSON. No extra text, no markdown fences.

Schema:
{
  "verdict": "pass|fail",
  "confidence": 0.0,
  "corrected": {
    "aspect": "string",
    "cause": "string",
    "inferred_attitude": "positive|negative|neutral|mixed",
    "verified_rationale": "string — corrected/verified explanation",
    "evidence_spans": [
      {"utt_id": "string", "char_start": 0, "char_end": 0}
    ]
  }
}"""

VERIFIER_USER_TEMPLATE = """Verify whether the generator's reasoning is supported by the text in the window.

Window:
{window_text}

Generator output:
{generator_json}
"""


def run_rvisa_verifier(
    generator_output: dict,
    window: List[PreprocessedUtterance],
    model_name: str = "mistral-small-latest",
    temperature: float = 0.1,
) -> Optional[dict]:
    """
    Step 2B: Run RVISA Verifier to check generator reasoning.
    Returns parsed JSON dict or None.
    """
    window_text = serialize_window(window)

    user_prompt = VERIFIER_USER_TEMPLATE.format(
        window_text=window_text,
        generator_json=json.dumps(generator_output, ensure_ascii=False),
    )

    try:
        response = ask(
            user_prompt=user_prompt,
            sys_prompt=VERIFIER_SYSTEM_PROMPT,
            model_name=model_name,
            temperature=temperature,
            infinite_retry=True,
        )
        return _try_parse_json(response)
    except Exception as e:
        logger.error("RVISA Verifier failed: %s", e)
        return None


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


# ========================= COMBINED STEP 2 =========================

def _validate_attitude(val: str) -> str:
    """Normalize attitude to valid enum."""
    val = val.lower().strip()
    valid = {'positive', 'negative', 'neutral', 'mixed'}
    return val if val in valid else 'neutral'


def run_rvisa(
    anchor_utt_id: str,
    preprocessed: PreprocessedConversation,
    transmistral_output: TransMistralOutput,
    model_name: str = "mistral-small-latest",
    k_before: int = 6,
    k_after: int = 2,
) -> RVISAOutput:
    """
    Step 2: Full RVISA pipeline — Generator + Verifier.
    
    Args:
        anchor_utt_id: ID of the anchor utterance.
        preprocessed: Step 0 output.
        transmistral_output: Step 1 output.
        
    Returns:
        RVISAOutput with verified reasoning or fail verdict.
    """
    conv_id = preprocessed.conversation_id

    # Find anchor utterance
    anchor_utt = None
    for utt in preprocessed.utterances:
        if utt.utt_id == anchor_utt_id:
            anchor_utt = utt
            break

    if anchor_utt is None:
        logger.error("Anchor utterance %s not found", anchor_utt_id)
        return RVISAOutput(
            conversation_id=conv_id, utt_id=anchor_utt_id,
            verdict=Verdict.FAIL, confidence=0.0,
        )

    # Build window
    window = build_rvisa_window(
        anchor_utt_id, preprocessed,
        transmistral_output.flip_events,
        k_before=k_before, k_after=k_after,
    )

    # 2A: Generator
    gen_result = run_rvisa_generator(
        anchor_utt, window,
        transmistral_output.context_summary,
        model_name=model_name,
    )

    if gen_result is None:
        logger.warning("RVISA Generator returned None for anchor %s", anchor_utt_id)
        return RVISAOutput(
            conversation_id=conv_id, utt_id=anchor_utt_id,
            verdict=Verdict.FAIL, confidence=0.0,
        )

    # 2B: Verifier
    ver_result = run_rvisa_verifier(gen_result, window, model_name=model_name)

    if ver_result is None:
        logger.warning("RVISA Verifier returned None for anchor %s", anchor_utt_id)
        return RVISAOutput(
            conversation_id=conv_id, utt_id=anchor_utt_id,
            aspect=gen_result.get('aspect', ''),
            cause=gen_result.get('cause', ''),
            inferred_attitude=_validate_attitude(gen_result.get('inferred_attitude', 'neutral')),
            verified_rationale=gen_result.get('rationale', ''),
            verdict=Verdict.FAIL,
            confidence=0.0,
        )

    # Build output from verifier
    verdict_str = ver_result.get('verdict', 'fail').lower()
    corrected = ver_result.get('corrected', {})

    evidence_spans = []
    for span in corrected.get('evidence_spans', []):
        evidence_spans.append(EvidenceSpan(
            utt_id=span.get('utt_id', ''),
            char_start=int(span.get('char_start', 0)),
            char_end=int(span.get('char_end', 0)),
        ))

    return RVISAOutput(
        conversation_id=conv_id,
        utt_id=anchor_utt_id,
        aspect=corrected.get('aspect', gen_result.get('aspect', '')),
        cause=corrected.get('cause', gen_result.get('cause', '')),
        inferred_attitude=_validate_attitude(
            corrected.get('inferred_attitude', gen_result.get('inferred_attitude', 'neutral'))
        ),
        verified_rationale=corrected.get('verified_rationale', gen_result.get('rationale', '')),
        evidence_spans=evidence_spans,
        verdict=Verdict.PASS if verdict_str == 'pass' else Verdict.FAIL,
        confidence=float(ver_result.get('confidence', 0.0)),
    )