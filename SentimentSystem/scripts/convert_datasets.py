"""
Convert ALL paper datasets -> ONE unified input JSON for the pipeline.

- TransMistral (MELD/MaSaC): each episode becomes a multi-utterance
  ConversationObject (keeps conversational context).
- RVISA (SemEval-2014): each review sentence -> 1-utterance ConversationObject.
- MASIVE (GoEmotions/EmoEvent): each text -> 1-utterance ConversationObject.

Output: data/converted/unified_input.json  (list of ConversationObjects)

Usage:
  python scripts/convert_datasets.py --limit 20
  python scripts/convert_datasets.py --limit 0        # all data
  python scripts/convert_datasets.py --dataset rvisa   # only RVISA
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = DATA / "converted"


# ──────────────────── TransMistral ────────────────────
# Keep full episode as multi-utterance conversation (has context)

def convert_transmistral(limit: int = 0) -> list[dict]:
    results = []
    files = [
        ("MELD_test_efr.json",  "en",    "MELD"),
        ("MaSaC_test_erc.json", "mixed", "MaSaC"),
    ]

    for fname, lang, tag in files:
        path = DATA / "transmistral" / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        with open(path, "r", encoding="utf-8") as f:
            episodes = json.load(f)
        print(f"  Reading {fname}: {len(episodes)} episodes")

        for ep_idx, ep in enumerate(episodes):
            if limit and len(results) >= limit:
                break

            ep_name  = ep.get("episode", f"ep_{ep_idx}")
            utts     = ep.get("utterances", [])
            speakers = ep.get("speakers", [])
            emotions = ep.get("emotions", ep.get("labels", []))

            # Build multi-utterance conversation
            utterances = []
            gt_emotions = []
            for i, text in enumerate(utts):
                speaker = speakers[i] if i < len(speakers) else f"speaker_{i}"
                emotion = emotions[i] if i < len(emotions) else "unknown"
                gt_emotions.append(emotion)
                utterances.append({
                    "utt_id": f"U{i+1}",
                    "speaker_id": speaker,
                    "timestamp": "",
                    "text_raw": text,
                    "reply_to_utt_id": f"U{i}" if i > 0 else None,
                })

            conv = {
                "conversation_id": f"{tag}_{ep_name}",
                "source": "chat",
                "language_hint": lang,
                "utterances": utterances,
                "meta": {
                    "dataset": f"transmistral_{tag}",
                    "has_context": True,
                    "ground_truth_emotions": gt_emotions,
                },
            }
            results.append(conv)

        if limit and len(results) >= limit:
            break

    return results


# ──────────────────── RVISA (SemEval XML) ────────────────────
# Each sentence -> 1-utterance conversation, no context

def convert_rvisa(limit: int = 0) -> list[dict]:
    results = []
    files = [
        ("laptops/Laptops_Test_Gold.xml",         "laptops"),
        ("restaurants/Restaurants_Test_Gold.xml",  "restaurants"),
    ]

    for rel_path, domain in files:
        path = DATA / "rvisa" / rel_path
        if not path.exists():
            print(f"  [SKIP] {rel_path} not found")
            continue

        tree = ET.parse(path)
        sentences = tree.getroot().findall(".//sentence")
        print(f"  Reading {rel_path}: {len(sentences)} sentences")

        for sent in sentences:
            if limit and len(results) >= limit:
                break

            sent_id = sent.get("id", "unknown")
            text_el = sent.find("text")
            text = text_el.text if text_el is not None else ""

            aspects = []
            at_el = sent.find("aspectTerms")
            if at_el is not None:
                for at in at_el.findall("aspectTerm"):
                    aspects.append({
                        "term": at.get("term", ""),
                        "polarity": at.get("polarity", ""),
                        "from": int(at.get("from", 0)),
                        "to": int(at.get("to", 0)),
                    })

            conv = {
                "conversation_id": f"RVISA_{domain}_{sent_id}",
                "source": "forum",
                "language_hint": "en",
                "utterances": [{
                    "utt_id": "U1",
                    "speaker_id": "reviewer",
                    "timestamp": "",
                    "text_raw": text,
                    "reply_to_utt_id": None,
                }],
                "meta": {
                    "dataset": f"rvisa_{domain}",
                    "has_context": False,
                    "ground_truth_aspects": aspects,
                },
            }
            results.append(conv)

    return results


# ──────────────────── MASIVE (CSV) ────────────────────
# Each text -> 1-utterance conversation, no context

def convert_masive(limit: int = 0) -> list[dict]:
    results = []
    files = [
        ("goemo_ekman_test.csv",   "GoEmo_Ekman", "en"),
        ("emo_event_en_test.csv",  "EmoEvent_EN", "en"),
    ]

    for fname, tag, lang in files:
        path = DATA / "masive" / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"  Reading {fname}: {len(rows)} texts")

        for row in rows:
            if limit and len(results) >= limit:
                break

            text_id = row.get("id", f"row_{len(results)}")
            text    = row.get("text", "")
            label   = row.get("label_txt", row.get("label", "unknown"))

            conv = {
                "conversation_id": f"MASIVE_{tag}_{text_id}",
                "source": "forum",
                "language_hint": lang,
                "utterances": [{
                    "utt_id": "U1",
                    "speaker_id": "user",
                    "timestamp": "",
                    "text_raw": text,
                    "reply_to_utt_id": None,
                }],
                "meta": {
                    "dataset": f"masive_{tag}",
                    "has_context": False,
                    "ground_truth_label": label,
                },
            }
            results.append(conv)

    return results


# ──────────────────── main ────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert paper datasets -> unified pipeline input"
    )
    parser.add_argument("--dataset", choices=["transmistral", "rvisa", "masive"],
                        help="Convert only one dataset (default: all)")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max items PER dataset (0 = all). Default: 20")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["transmistral", "rvisa", "masive"]
    all_convs = []

    for ds in datasets:
        print(f"\n=== {ds.upper()} ===")
        if ds == "transmistral":
            convs = convert_transmistral(args.limit)
        elif ds == "rvisa":
            convs = convert_rvisa(args.limit)
        elif ds == "masive":
            convs = convert_masive(args.limit)
        else:
            continue
        print(f"  -> {len(convs)} conversations")
        all_convs.extend(convs)

    # Save unified input
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "unified_input.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_convs, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Total: {len(all_convs)} conversations")
    print(f"Saved: {out_path}")
    print(f"\nNext: python scripts/run_batch.py --input {out_path} -v")


if __name__ == "__main__":
    main()
