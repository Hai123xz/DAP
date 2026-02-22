"""
Step 1 — TransMistral Contextual Parsing (API-Based)

Given a preprocessed conversation, produces:
  - Coarse emotion per utterance
  - Emotion flips
  - Anchor utterances (requiring deeper reasoning)
  - Global context summary
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from call_llm_mistral import ask

from modules.models import (
    PreprocessedConversation,
    TransMistralOutput,
    CoarseTimelineEntry,
    AnchorEntry,
    FlipEvent,
    CoarseEmotion,
)

logger = logging.getLogger(__name__)


# ========================= CONVERSATION SERIALIZATION =========================

def serialize_conversation(conv: PreprocessedConversation, max_utterances: int = 200) -> str:
    """
    Serialize conversation to the stable format recommended by README.
    One utterance per line:
      [utt_id=U12][t=...][S=S2][reply_to=U9] raw:"..." | translated:"..."
    """
    lines = []
    utts = conv.utterances[-max_utterances:]  # tail window if too long

    for utt in utts:
        parts = [f'[utt_id={utt.utt_id}]']
        if utt.timestamp:
            parts.append(f'[t={utt.timestamp}]')
        parts.append(f'[S={utt.speaker_id}]')
        if utt.reply_to_utt_id:
            parts.append(f'[reply_to={utt.reply_to_utt_id}]')

        text_part = f'raw:"{utt.text_clean}"'
        if utt.text_translated:
            text_part += f' | translated:"{utt.text_translated}"'

        parts.append(text_part)
        lines.append(' '.join(parts))

    return '\n'.join(lines)


# ========================= PROMPT =========================

TRANSMISTRAL_SYSTEM_PROMPT = """You are a conversation context engine. Your task is to analyze conversational text and produce structured emotional analysis.

Return ONLY valid JSON matching the schema below. No extra text, no markdown fences, no explanation.

Schema:
{
  "context_summary": "string — brief summary of the conversation context and emotional dynamics",
  "coarse_timeline": [
    {
      "utt_id": "string",
      "coarse_emotion": "neutral|joy|sadness|anger|fear|disgust|surprise|mixed|unknown",
      "prob": 0.0,
      "flip_flag": false,
      "flip_type": "none"
    }
  ],
  "anchors": [
    {
      "utt_id": "string",
      "anchor_score": 0.0,
      "anchor_reason": "flip-trigger|high-arousal|sarcasm-marker|low-confidence|domain-escalation"
    }
  ],
  "flip_events": [
    {
      "from_utt_id": "string",
      "to_utt_id": "string",
      "flip_type": "pos->neg|neg->pos|calm->anger|...",
      "trigger_utt_id": "string"
    }
  ]
}"""

TRANSMISTRAL_USER_TEMPLATE = """Tasks:
1) Assign a coarse emotion to each utterance from: neutral, joy, sadness, anger, fear, disgust, surprise, mixed, unknown. Include probability.
2) Detect emotion flips — where the emotional trajectory changes. Identify which utterance triggers it.
3) Select anchor utterances that require deeper reasoning. Anchors are: flip triggers, high-arousal emotions, sarcasm markers, low-confidence predictions, or domain escalation patterns.

Conversation:
{conversation_text}"""


# ========================= JSON PARSING =========================

def _try_parse_json(text: str) -> Optional[dict]:
    """Try to parse JSON from model output, with repair attempts."""
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        lines = [l for l in lines if not l.strip().startswith('```')]
        text = '\n'.join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _validate_coarse_emotion(val: str) -> str:
    """Normalize coarse emotion string to valid enum."""
    val = val.lower().strip()
    valid = {'neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'mixed', 'unknown'}
    return val if val in valid else 'unknown'


# ========================= MAIN FUNCTION =========================

def run_transmistral(
    preprocessed: PreprocessedConversation,
    model_name: str = "mistral-small-latest",
    max_tokens: int = 16384,
    temperature: float = 0.1,
    max_retries: int = 2,
) -> TransMistralOutput:
    """
    Step 1: Run TransMistral contextual parsing via Mistral API.
    
    Args:
        preprocessed: Output from Step 0.
        model_name: Mistral model to use.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        max_retries: Number of retries on JSON parse failure.
        
    Returns:
        TransMistralOutput with coarse emotions, flips, and anchors.
    """
    conversation_text = serialize_conversation(preprocessed)
    user_prompt = TRANSMISTRAL_USER_TEMPLATE.format(conversation_text=conversation_text)

    # Try API call with retries
    parsed = None
    for attempt in range(max_retries + 1):
        try:
            response = ask(
                user_prompt=user_prompt,
                sys_prompt=TRANSMISTRAL_SYSTEM_PROMPT,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                infinite_retry=True,
            )

            parsed = _try_parse_json(response)
            if parsed is not None:
                break

            logger.warning(
                "Step 1 attempt %d: JSON parse failed, retrying with stricter prompt",
                attempt + 1,
            )
            # On retry, add stricter instruction
            user_prompt = "CRITICAL: Return ONLY valid JSON, no other text.\n\n" + user_prompt

        except Exception as e:
            logger.error("Step 1 attempt %d failed: %s", attempt + 1, e)

    # Build output from parsed JSON or empty fallback
    if parsed is None:
        logger.error("Step 1: All retries exhausted, returning empty output")
        return TransMistralOutput(conversation_id=preprocessed.conversation_id)

    # Parse timeline
    coarse_timeline = []
    for entry in parsed.get('coarse_timeline', []):
        coarse_timeline.append(CoarseTimelineEntry(
            utt_id=entry.get('utt_id', ''),
            coarse_emotion=_validate_coarse_emotion(entry.get('coarse_emotion', 'unknown')),
            prob=float(entry.get('prob', 0.0)),
            flip_flag=bool(entry.get('flip_flag', False)),
            flip_type=entry.get('flip_type', 'none'),
        ))

    # Parse anchors
    anchors = []
    for entry in parsed.get('anchors', []):
        anchors.append(AnchorEntry(
            utt_id=entry.get('utt_id', ''),
            anchor_score=float(entry.get('anchor_score', 0.0)),
            anchor_reason=entry.get('anchor_reason', ''),
        ))

    # Parse flip events
    flip_events = []
    for entry in parsed.get('flip_events', []):
        flip_events.append(FlipEvent(
            from_utt_id=entry.get('from_utt_id', ''),
            to_utt_id=entry.get('to_utt_id', ''),
            flip_type=entry.get('flip_type', ''),
            trigger_utt_id=entry.get('trigger_utt_id', ''),
        ))

    return TransMistralOutput(
        conversation_id=preprocessed.conversation_id,
        context_summary=parsed.get('context_summary', ''),
        coarse_timeline=coarse_timeline,
        anchors=anchors,
        flip_events=flip_events,
    )
