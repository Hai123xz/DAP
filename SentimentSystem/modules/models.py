"""
Shared Pydantic Data Contracts for the Sentiment Analysis Pipeline.

Implements Contracts A–E from the README specification:
  A — ConversationObject (raw input)
  B — PreprocessedConversation (Step 0 output)
  C — TransMistralOutput (Step 1 output)
  D — RVISAOutput (Step 2 output)
  E — MASIVEOutput (Step 3 output)
  Final — FinalEmotionReport
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ========================= ENUMS =========================

class SourceType(str, Enum):
    FACEBOOK = "facebook"
    CHAT = "chat"
    FORUM = "forum"


class LanguageTag(str, Enum):
    VI = "vi"
    EN = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CoarseEmotion(str, Enum):
    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class FlipType(str, Enum):
    POS_TO_NEG = "pos->neg"
    NEG_TO_POS = "neg->pos"
    CALM_TO_ANGER = "calm->anger"
    NONE = "none"


class InferredAttitude(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class NormalizationMethod(str, Enum):
    EXACT = "exact"
    EMBEDDING_MATCH = "embedding_match"
    LLM_RERANK = "llm_rerank"
    CLUSTER = "cluster"


class AnchorReason(str, Enum):
    FLIP_TRIGGER = "flip-trigger"
    HIGH_AROUSAL = "high-arousal"
    SARCASM_MARKER = "sarcasm-marker"
    LOW_CONFIDENCE = "low-confidence"
    DOMAIN_ESCALATION = "domain-escalation"


# ========================= CONTRACT A — Raw Input =========================

class RawUtterance(BaseModel):
    utt_id: str
    speaker_id: str
    timestamp: str = ""
    text_raw: str
    reply_to_utt_id: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ConversationObject(BaseModel):
    """Contract A — Raw conversation input from FB/chat/forum."""
    conversation_id: str
    source: SourceType = SourceType.CHAT
    language_hint: LanguageTag = LanguageTag.UNKNOWN
    utterances: List[RawUtterance]


# ========================= CONTRACT B — Preprocessed =========================

class PreprocessedUtterance(BaseModel):
    utt_id: str
    speaker_id: str
    timestamp: str = ""
    text_raw: str
    text_clean: str
    lang: LanguageTag = LanguageTag.UNKNOWN
    text_translated: Optional[str] = None
    reply_to_utt_id: Optional[str] = None


class PreprocessMeta(BaseModel):
    pivot_language: str = "en"
    translation_provider: str = "google"
    emoji_preserved: bool = True
    pii_redaction: bool = True


class PreprocessedConversation(BaseModel):
    """Contract B — Step 0 output, Step 1 input."""
    conversation_id: str
    utterances: List[PreprocessedUtterance]
    preprocess_meta: PreprocessMeta = Field(default_factory=PreprocessMeta)


# ========================= CONTRACT C — TransMistral =========================

class CoarseTimelineEntry(BaseModel):
    utt_id: str
    coarse_emotion: CoarseEmotion = CoarseEmotion.UNKNOWN
    prob: float = 0.0
    flip_flag: bool = False
    flip_type: str = "none"


class AnchorEntry(BaseModel):
    utt_id: str
    anchor_score: float = 0.0
    anchor_reason: str = ""


class FlipEvent(BaseModel):
    from_utt_id: str
    to_utt_id: str
    flip_type: str
    trigger_utt_id: str


class TransMistralOutput(BaseModel):
    """Contract C — Step 1 output."""
    conversation_id: str
    context_summary: str = ""
    context_vector: List[float] = Field(default_factory=list)
    coarse_timeline: List[CoarseTimelineEntry] = Field(default_factory=list)
    anchors: List[AnchorEntry] = Field(default_factory=list)
    flip_events: List[FlipEvent] = Field(default_factory=list)


# ========================= CONTRACT D — RVISA =========================

class EvidenceSpan(BaseModel):
    utt_id: str
    char_start: int = 0
    char_end: int = 0


class RVISAOutput(BaseModel):
    """Contract D — Step 2 output."""
    conversation_id: str
    utt_id: str
    aspect: str = ""
    cause: str = ""
    inferred_attitude: InferredAttitude = InferredAttitude.NEUTRAL
    verified_rationale: str = ""
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)
    verdict: Verdict = Verdict.FAIL
    confidence: float = 0.0


# ========================= CONTRACT E — MASIVE =========================

class NormalizationInfo(BaseModel):
    method: NormalizationMethod = NormalizationMethod.EXACT
    matched_vocab_id: Optional[str] = None
    cluster_id: Optional[str] = None


class MASIVEOutput(BaseModel):
    """Contract E — Step 3 output."""
    conversation_id: str
    utt_id: str
    fine_grained_label: str = ""
    alt_labels: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    normalization: NormalizationInfo = Field(default_factory=NormalizationInfo)


# ========================= FINAL REPORT =========================

class CoarseBlock(BaseModel):
    emotion: str = "unknown"
    prob: float = 0.0
    flip_flag: bool = False
    flip_type: str = "none"


class AnchorBlock(BaseModel):
    is_anchor: bool = False
    score: float = 0.0
    reason: str = ""


class RVISABlock(BaseModel):
    verdict: str = ""
    confidence: float = 0.0
    aspect: str = ""
    cause: str = ""
    verified_rationale: str = ""
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)


class MASIVEBlock(BaseModel):
    label: str = ""
    alt: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    normalization: NormalizationInfo = Field(default_factory=NormalizationInfo)


class TimelineEntry(BaseModel):
    utt_id: str
    speaker_id: str
    timestamp: str = ""
    text_preview: str = ""
    coarse: CoarseBlock = Field(default_factory=CoarseBlock)
    anchor: AnchorBlock = Field(default_factory=AnchorBlock)
    rvisa: Optional[RVISABlock] = None
    masive: Optional[MASIVEBlock] = None


class ModelVersions(BaseModel):
    step0: str = "preprocess-v1"
    step1: str = "mistral-small-latest"
    step2: str = "mistral-small-latest"
    step3: str = "mistral-small-latest"


class PipelineOptions(BaseModel):
    max_anchors: int = 20
    window_before: int = 6
    window_after: int = 2


class ReportMeta(BaseModel):
    model_versions: ModelVersions = Field(default_factory=ModelVersions)
    options: PipelineOptions = Field(default_factory=PipelineOptions)


class FinalEmotionReport(BaseModel):
    """Final output of the entire pipeline."""
    conversation_id: str
    context_summary: str = ""
    timeline: List[TimelineEntry] = Field(default_factory=list)
    key_anchors: List[str] = Field(default_factory=list)
    dominant_states: List[str] = Field(default_factory=list)
    meta: ReportMeta = Field(default_factory=ReportMeta)
