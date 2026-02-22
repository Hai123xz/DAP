"""
Orchestrator — Full End-to-End Sentiment Analysis Pipeline

Algorithm:
  1. Receive raw ConversationObject
  2. Step 0: Preprocess (flatten, clean, detect language, translate, redact PII)
  3. Step 1: TransMistral contextual parsing (coarse emotions + flips + anchors)
  4. Anchor gating (threshold + max_anchors)
  5. For each anchor:
     a. Step 2: RVISA Generator → Verifier
     b. If PASS: Step 3: MASIVE fine-grained labeling
     c. If FAIL: fallback to coarse emotion
  6. Assemble FinalEmotionReport
  7. Output report as JSON
"""

from __future__ import annotations

import json
import logging
import sys
import os
import argparse
from typing import List, Optional
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.models import (
    ConversationObject,
    RawUtterance,
    FinalEmotionReport,
    TimelineEntry,
    CoarseBlock,
    AnchorBlock,
    RVISABlock,
    MASIVEBlock,
    ReportMeta,
    ModelVersions,
    PipelineOptions,
    EvidenceSpan,
    NormalizationInfo,
    Verdict,
)
from modules.preprocessing.engine import preprocess_conversation
from modules.transmistral.engine import run_transmistral
from modules.rvisa.engine import run_rvisa
from modules.masive.engine import run_masive

logger = logging.getLogger(__name__)


# ========================= CONFIG =========================

def load_config(config_path: str = None) -> dict:
    """Load pipeline config from YAML file or return defaults."""
    defaults = {
        'model_name': 'mistral-small-latest',
        'anchor_threshold': 0.65,
        'max_anchors': 20,
        'window_before': 6,
        'window_after': 2,
        'enable_translation': False,
        'enable_pii_redaction': True,
        'pivot_language': 'en',
        'always_include_flip_triggers': True,
    }

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            if user_config:
                defaults.update(user_config)
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", config_path, e)

    return defaults


# ========================= ANCHOR GATING =========================

def gate_anchors(
    anchors: list,
    flip_events: list,
    threshold: float = 0.65,
    max_anchors: int = 20,
    always_include_flip_triggers: bool = True,
) -> list:
    """
    Anchor gating — primary cost/latency control.
    
    Rules:
      - Keep anchors with anchor_score >= threshold
      - Always include flip triggers (even if below threshold)
      - Cap to top max_anchors by score
    """
    # Collect flip trigger IDs
    flip_trigger_ids = set()
    if always_include_flip_triggers:
        for flip in flip_events:
            flip_trigger_ids.add(flip.trigger_utt_id)

    # Filter
    gated = []
    for anchor in anchors:
        if anchor.anchor_score >= threshold:
            gated.append(anchor)
        elif anchor.utt_id in flip_trigger_ids:
            gated.append(anchor)

    # Sort by score descending, cap
    gated.sort(key=lambda a: a.anchor_score, reverse=True)
    return gated[:max_anchors]


# ========================= FINAL ASSEMBLY =========================

def assemble_report(
    conversation_id: str,
    preprocessed,
    transmistral_output,
    rvisa_results: dict,
    masive_results: dict,
    config: dict,
) -> FinalEmotionReport:
    """
    Assemble final emotion report from all step outputs.
    """
    # Build lookup maps
    coarse_map = {e.utt_id: e for e in transmistral_output.coarse_timeline}
    anchor_map = {a.utt_id: a for a in transmistral_output.anchors}

    timeline = []
    for utt in preprocessed.utterances:
        uid = utt.utt_id
        coarse_entry = coarse_map.get(uid)
        anchor_entry = anchor_map.get(uid)

        entry = TimelineEntry(
            utt_id=uid,
            speaker_id=utt.speaker_id,
            timestamp=utt.timestamp,
            text_preview=utt.text_clean[:100],
            coarse=CoarseBlock(
                emotion=coarse_entry.coarse_emotion if coarse_entry else 'unknown',
                prob=coarse_entry.prob if coarse_entry else 0.0,
                flip_flag=coarse_entry.flip_flag if coarse_entry else False,
                flip_type=coarse_entry.flip_type if coarse_entry else 'none',
            ),
            anchor=AnchorBlock(
                is_anchor=uid in anchor_map,
                score=anchor_entry.anchor_score if anchor_entry else 0.0,
                reason=anchor_entry.anchor_reason if anchor_entry else '',
            ),
        )

        # Attach RVISA result if exists
        if uid in rvisa_results:
            rv = rvisa_results[uid]
            entry.rvisa = RVISABlock(
                verdict=rv.verdict,
                confidence=rv.confidence,
                aspect=rv.aspect,
                cause=rv.cause,
                verified_rationale=rv.verified_rationale,
                evidence_spans=rv.evidence_spans,
            )

        # Attach MASIVE result if exists
        if uid in masive_results:
            mv = masive_results[uid]
            entry.masive = MASIVEBlock(
                label=mv.fine_grained_label,
                alt=mv.alt_labels,
                confidence=mv.confidence,
                normalization=mv.normalization,
            )

        timeline.append(entry)

    # Compute aggregates
    key_anchors = sorted(
        [uid for uid in rvisa_results if rvisa_results[uid].verdict == Verdict.PASS],
        key=lambda uid: rvisa_results[uid].confidence,
        reverse=True,
    )[:5]

    # Dominant states from MASIVE results
    state_counts = Counter()
    for mv in masive_results.values():
        if mv.fine_grained_label and mv.fine_grained_label != 'unknown':
            state_counts[mv.fine_grained_label] += 1
    dominant_states = [s for s, _ in state_counts.most_common(3)]

    return FinalEmotionReport(
        conversation_id=conversation_id,
        context_summary=transmistral_output.context_summary,
        timeline=timeline,
        key_anchors=key_anchors,
        dominant_states=dominant_states,
        meta=ReportMeta(
            model_versions=ModelVersions(
                step0='preprocess-v1',
                step1=config.get('model_name', 'mistral-small-latest'),
                step2=config.get('model_name', 'mistral-small-latest'),
                step3=config.get('model_name', 'mistral-small-latest'),
            ),
            options=PipelineOptions(
                max_anchors=config.get('max_anchors', 20),
                window_before=config.get('window_before', 6),
                window_after=config.get('window_after', 2),
            ),
        ),
    )


# ========================= MAIN PIPELINE =========================

def run_pipeline(
    raw: ConversationObject,
    config: dict = None,
) -> FinalEmotionReport:
    """
    Run the full end-to-end sentiment analysis pipeline.
    
    Args:
        raw: Raw ConversationObject input.
        config: Pipeline configuration dict.
        
    Returns:
        FinalEmotionReport with full timeline and analysis.
    """
    if config is None:
        config = load_config()

    conv_id = raw.conversation_id
    model_name = config.get('model_name', 'mistral-small-latest')

    # ---- Step 0: Preprocess ----
    logger.info("[%s] Step 0: Preprocessing...", conv_id)
    preprocessed = preprocess_conversation(
        raw,
        enable_translation=config.get('enable_translation', False),
        enable_pii_redaction=config.get('enable_pii_redaction', True),
        pivot_language=config.get('pivot_language', 'en'),
    )
    logger.info("[%s] Step 0 done: %d utterances", conv_id, len(preprocessed.utterances))

    # ---- Step 1: TransMistral ----
    logger.info("[%s] Step 1: TransMistral contextual parsing...", conv_id)
    transmistral_output = run_transmistral(
        preprocessed,
        model_name=model_name,
    )
    logger.info(
        "[%s] Step 1 done: %d timeline entries, %d anchors, %d flips",
        conv_id,
        len(transmistral_output.coarse_timeline),
        len(transmistral_output.anchors),
        len(transmistral_output.flip_events),
    )

    # ---- Anchor Gating ----
    gated_anchors = gate_anchors(
        transmistral_output.anchors,
        transmistral_output.flip_events,
        threshold=config.get('anchor_threshold', 0.65),
        max_anchors=config.get('max_anchors', 20),
        always_include_flip_triggers=config.get('always_include_flip_triggers', True),
    )
    logger.info("[%s] Anchor gating: %d → %d anchors",
                conv_id, len(transmistral_output.anchors), len(gated_anchors))

    # ---- Step 2 + 3: RVISA + MASIVE per anchor ----
    rvisa_results = {}
    masive_results = {}

    for i, anchor in enumerate(gated_anchors):
        logger.info("[%s] Processing anchor %d/%d: %s",
                    conv_id, i + 1, len(gated_anchors), anchor.utt_id)

        # Step 2: RVISA
        rvisa_output = run_rvisa(
            anchor.utt_id,
            preprocessed,
            transmistral_output,
            model_name=model_name,
            k_before=config.get('window_before', 6),
            k_after=config.get('window_after', 2),
        )
        rvisa_results[anchor.utt_id] = rvisa_output

        # Step 3: MASIVE (only if RVISA passed)
        if rvisa_output.verdict == Verdict.PASS:
            masive_output = run_masive(rvisa_output, model_name=model_name)
            masive_results[anchor.utt_id] = masive_output
            logger.info("[%s] Anchor %s: PASS → label='%s' (%.2f)",
                        conv_id, anchor.utt_id,
                        masive_output.fine_grained_label, masive_output.confidence)
        else:
            logger.info("[%s] Anchor %s: FAIL (confidence=%.2f), skipping MASIVE",
                        conv_id, anchor.utt_id, rvisa_output.confidence)

    # ---- Final Assembly ----
    logger.info("[%s] Assembling final report...", conv_id)
    report = assemble_report(
        conv_id, preprocessed, transmistral_output,
        rvisa_results, masive_results, config,
    )
    logger.info("[%s] Pipeline complete. %d dominant states: %s",
                conv_id, len(report.dominant_states), report.dominant_states)

    return report


# ========================= CLI =========================

def main():
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis Pipeline — TransMistral + RVISA + MASIVE'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Path to input JSON file (ConversationObject format)'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Path to output JSON file (FinalEmotionReport). Defaults to stdout.'
    )
    parser.add_argument(
        '-c', '--config', default=None,
        help='Path to pipeline config YAML file'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    # Load config
    config = load_config(args.config)

    # Load input
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    raw = ConversationObject(**raw_data)

    # Run pipeline
    report = run_pipeline(raw, config)

    # Output
    report_json = report.model_dump_json(indent=2)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report_json)
        logger.info("Report saved to %s", args.output)
    else:
        print(report_json)


if __name__ == '__main__':
    main()
