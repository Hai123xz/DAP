"""
Evaluation Runner - Feed eval_dataset.jsonl through the full pipeline.

Reads JSONL samples, wraps each into ConversationObject,
runs through the full pipeline (Step 0->1->2->3),
and outputs eval_results.jsonl with true_label vs predicted_label.

Usage:
  python scripts/run_batch.py --input data/eval_dataset.jsonl --limit 10 -v
  python scripts/run_batch.py --input data/eval_dataset.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.models import ConversationObject
from scripts.run_pipeline import run_pipeline, load_config

logger = logging.getLogger(__name__)


def sample_to_conversation(sample: dict) -> ConversationObject:
    """
    Wrap an eval sample into a ConversationObject for the pipeline.

    - If context is present: parse "[Speaker]: text" lines into utterances,
      then append the main text as the last utterance.
    - If no context: single-utterance conversation.
    """
    utterances = []
    context = sample.get("context", "")

    if context.strip():
        # Parse context lines: "[Speaker]: text"
        ctx_lines = context.split("\n")
        for i, line in enumerate(ctx_lines):
            m = re.match(r"\[(.+?)\]:\s*(.*)", line)
            if m:
                speaker, text = m.group(1), m.group(2)
            else:
                speaker, text = f"ctx_{i}", line

            utterances.append({
                "utt_id": f"C{i+1}",
                "speaker_id": speaker,
                "timestamp": "",
                "text_raw": text,
                "reply_to_utt_id": f"C{i}" if i > 0 else None,
            })

    # Add the main text as the target utterance
    main_utt_id = f"U{len(utterances)+1}"
    utterances.append({
        "utt_id": main_utt_id,
        "speaker_id": "target_speaker",
        "timestamp": "",
        "text_raw": sample["text"],
        "reply_to_utt_id": utterances[-1]["utt_id"] if utterances else None,
    })

    conv = ConversationObject(
        conversation_id=f"eval_{sample['id']}",
        source="chat" if context.strip() else "forum",
        language_hint="en",
        utterances=utterances,
    )
    return conv, main_utt_id


def extract_predicted_label(report_dict: dict, target_utt_id: str) -> tuple[str, str, float]:
    """
    Extract the predicted label for the target utterance from the pipeline report.

    Returns: (predicted_label, coarse_emotion, confidence)
    """
    timeline = report_dict.get("timeline", [])

    # Find the target utterance in timeline
    target_entry = None
    for entry in timeline:
        if entry.get("utt_id") == target_utt_id:
            target_entry = entry
            break

    # Fallback: use last utterance
    if target_entry is None and timeline:
        target_entry = timeline[-1]

    if target_entry is None:
        return "unknown", "unknown", 0.0

    coarse = target_entry.get("coarse", {})
    coarse_emotion = coarse.get("emotion", "unknown")
    confidence = coarse.get("prob", 0.0)

    # If MASIVE gave a fine-grained label, use that
    masive = target_entry.get("masive")
    if masive and masive.get("label"):
        return masive["label"], coarse_emotion, masive.get("confidence", confidence)

    return coarse_emotion, coarse_emotion, confidence


def process_one(sample: dict, config: dict, verbose: bool) -> dict:
    """Process a single sample through the pipeline. Thread-safe."""
    sid = sample["id"]
    ds = sample["dataset"]
    try:
        conv, target_utt = sample_to_conversation(sample)
        start = time.time()
        report = run_pipeline(conv, config)
        elapsed = time.time() - start

        report_dict = json.loads(report.model_dump_json())
        predicted, coarse, conf = extract_predicted_label(report_dict, target_utt)
        is_match = predicted.lower().strip() == sample["true_label"].lower().strip()

        return {
            "id": sid,
            "text": sample["text"],
            "context": sample.get("context", ""),
            "dataset": ds,
            "true_label": sample["true_label"],
            "predicted_label": predicted,
            "coarse_emotion": coarse,
            "confidence": round(conf, 3),
            "match": is_match,
            "time_s": round(elapsed, 1),
            "error": None,
        }
    except Exception as e:
        if verbose:
            logger.error("FAILED id=%s: %s", sid, e, exc_info=True)
        return {
            "id": sid,
            "text": sample["text"],
            "context": sample.get("context", ""),
            "dataset": ds,
            "true_label": sample["true_label"],
            "predicted_label": "ERROR",
            "coarse_emotion": "",
            "confidence": 0.0,
            "match": False,
            "time_s": 0,
            "error": str(e),
        }


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    parser = argparse.ArgumentParser(
        description="Run eval_dataset.jsonl through the full pipeline"
    )
    parser.add_argument("--input", "-i", default="data/eval_dataset.jsonl",
                        help="Path to eval JSONL file")
    parser.add_argument("--output", "-o", default="data/results/eval_results.jsonl",
                        help="Output JSONL with results")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from sample index (for resuming)")
    parser.add_argument("--workers", "-w", type=int, default=10,
                        help="Number of concurrent workers (default: 10)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Load samples
    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    # Apply start/limit
    samples = all_samples[args.start:]
    if args.limit:
        samples = samples[:args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Evaluation Pipeline Runner")
    print(f"  Input:   {input_path} ({len(all_samples)} total)")
    print(f"  Range:   [{args.start} .. {args.start + len(samples)})")
    print(f"  Workers: {args.workers}")
    print(f"  Output:  {output_path}")
    print("=" * 60)

    success = 0
    failed = 0
    match_count = 0
    total_time = time.time()
    write_lock = Lock()

    mode = "a" if args.start > 0 else "w"
    with open(output_path, mode, encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, s, config, args.verbose): s
                for s in samples
            }

            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                sid = result["id"]
                ds = result["dataset"]

                with write_lock:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()

                if result["error"]:
                    failed += 1
                    print(f"  [{done_count}/{len(samples)}] id={sid} ({ds}) "
                          f"ERROR: {result['error'][:60]}")
                else:
                    success += 1
                    if result["match"]:
                        match_count += 1
                    tag = "OK" if result["match"] else "MISS"
                    print(f"  [{done_count}/{len(samples)}] id={sid} ({ds}) "
                          f"[{tag}] true={result['true_label']} | "
                          f"pred={result['predicted_label']} | "
                          f"{result['time_s']}s")

    elapsed = time.time() - total_time
    accuracy = (match_count / success * 100) if success > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"  DONE: {success} success, {failed} failed")
    print(f"  Accuracy: {match_count}/{success} = {accuracy:.1f}%")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Results: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

