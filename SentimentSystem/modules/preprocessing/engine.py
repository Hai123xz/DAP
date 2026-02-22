"""
Step 0 — Ingestion + Preprocess + Normalization

Transforms raw ConversationObject into PreprocessedConversation:
  0.1  Thread flattening (tree → linear timeline)
  0.2  Speaker canonicalization
  0.3  Text cleaning (preserve emotion signals)
  0.4  PII redaction
  0.5  Language detection
  0.6  Translation view (optional)
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, Optional

from modules.models import (
    ConversationObject,
    PreprocessedConversation,
    PreprocessedUtterance,
    PreprocessMeta,
    LanguageTag,
)

logger = logging.getLogger(__name__)


# ========================= 0.1 Thread Flattening =========================

def flatten_thread(utterances: list) -> list:
    """
    Sort utterances into a time-ordered linear list.
    Preserves reply_to_utt_id for downstream models.
    """
    def _sort_key(u):
        ts = u.timestamp if hasattr(u, 'timestamp') else u.get('timestamp', '')
        return ts or ''

    sorted_utts = sorted(utterances, key=_sort_key)
    return sorted_utts


# ========================= 0.2 Speaker Canonicalization =========================

def canonicalize_speakers(utterances: list) -> Dict[str, str]:
    """
    Build a stable speaker_id mapping.
    Returns: {original_id: canonical_id}
    """
    seen = {}
    counter = 1
    for utt in utterances:
        sid = utt.speaker_id if hasattr(utt, 'speaker_id') else utt.get('speaker_id', '')
        if sid and sid not in seen:
            seen[sid] = f"S{counter}"
            counter += 1
    return seen


# ========================= 0.3 Text Cleaning =========================

# PII patterns
_PII_PATTERNS = {
    'phone': re.compile(
        r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
    ),
    'email': re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    ),
    'url': re.compile(
        r'https?://\S+|www\.\S+'
    ),
}

# Characters to keep for emotion signals
_MENTION_PATTERN = re.compile(r'@\w+')


def clean_text(text: str) -> str:
    """
    Clean text while preserving emotion signals.
    
    DO:
      - Replace URLs → <url>
      - Replace @mentions → <user>
      - Normalize whitespace
    
    DO NOT:
      - Remove emojis
      - Strip "!!!" or "???"
      - Aggressively normalize repeated characters
    """
    # Replace URLs
    text = _PII_PATTERNS['url'].sub('<url>', text)

    # Replace @mentions
    text = _MENTION_PATTERN.sub('<user>', text)

    # Normalize whitespace (but keep newlines as spaces)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ========================= 0.4 PII Redaction =========================

def redact_pii(text: str) -> str:
    """
    Redact personally identifiable information before sending to
    third-party model APIs.
    """
    text = _PII_PATTERNS['phone'].sub('<pii_phone>', text)
    text = _PII_PATTERNS['email'].sub('<pii_email>', text)
    return text


# ========================= 0.5 Language Detection =========================

def detect_language(text: str) -> LanguageTag:
    """
    Detect language of a text utterance.
    Returns LanguageTag enum.
    """
    try:
        from langdetect import detect
        lang = detect(text)
        if lang == 'vi':
            return LanguageTag.VI
        elif lang == 'en':
            return LanguageTag.EN
        else:
            # Check for mixed VI/EN patterns
            vi_chars = len(re.findall(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text.lower()))
            if vi_chars > 0:
                return LanguageTag.MIXED
            return LanguageTag.UNKNOWN
    except Exception:
        # Fallback: simple heuristic
        vi_chars = len(re.findall(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text.lower()))
        en_chars = len(re.findall(r'[a-z]', text.lower()))
        if vi_chars > 0 and en_chars > 0:
            return LanguageTag.MIXED
        elif vi_chars > 0:
            return LanguageTag.VI
        elif en_chars > 0:
            return LanguageTag.EN
        return LanguageTag.UNKNOWN


# ========================= 0.6 Translation =========================

def translate_text(text: str, source_lang: str = 'auto', target_lang: str = 'en') -> Optional[str]:
    """
    Translate text to pivot language using deep-translator.
    Returns translated text or None on failure.
    """
    if not text or len(text.strip()) < 2:
        return None

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated
    except Exception as e:
        logger.warning("Translation failed for text '%s...': %s", text[:50], e)
        return None


# ========================= MAIN FUNCTION =========================

def preprocess_conversation(
    raw: ConversationObject,
    enable_translation: bool = False,
    enable_pii_redaction: bool = True,
    pivot_language: str = "en",
) -> PreprocessedConversation:
    """
    Step 0: Transform raw ConversationObject into PreprocessedConversation.
    
    This is the main entry point for preprocessing.
    """
    # 0.1 Flatten thread
    sorted_utts = flatten_thread(raw.utterances)

    # 0.2 Canonicalize speakers
    speaker_map = canonicalize_speakers(sorted_utts)

    # Process each utterance
    processed_utts = []
    for utt in sorted_utts:
        # 0.3 Clean text
        text_clean = clean_text(utt.text_raw)

        # 0.4 PII redaction
        if enable_pii_redaction:
            text_clean = redact_pii(text_clean)

        # 0.5 Language detection
        lang = detect_language(utt.text_raw)

        # 0.6 Translation (optional)
        text_translated = None
        if enable_translation and lang in (LanguageTag.VI, LanguageTag.MIXED):
            text_translated = translate_text(text_clean, target_lang=pivot_language)

        processed_utts.append(PreprocessedUtterance(
            utt_id=utt.utt_id,
            speaker_id=speaker_map.get(utt.speaker_id, utt.speaker_id),
            timestamp=utt.timestamp,
            text_raw=utt.text_raw,
            text_clean=text_clean,
            lang=lang,
            text_translated=text_translated,
            reply_to_utt_id=utt.reply_to_utt_id,
        ))

    return PreprocessedConversation(
        conversation_id=raw.conversation_id,
        utterances=processed_utts,
        preprocess_meta=PreprocessMeta(
            pivot_language=pivot_language,
            translation_provider="google" if enable_translation else "none",
            emoji_preserved=True,
            pii_redaction=enable_pii_redaction,
        ),
    )
