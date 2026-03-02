"""
Build evaluation dataset from paper data -> JSONL format.

Each line = 1 sample:
  {"id", "text", "context", "true_label", "dataset"}

- TransMistral (MELD/MaSaC): each utterance = 1 sample,
  context = surrounding utterances in the episode.
- RVISA (SemEval): each sentence = 1 sample, context = "".
- MASIVE (GoEmotions/EmoEvent): each text = 1 sample, context = "".

Output: data/eval_dataset.jsonl

Usage:
  python scripts/build_eval_dataset.py --limit 20
  python scripts/build_eval_dataset.py --limit 0   # all data
"""

from __future__ import annotations
import argparse, csv, json, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ══════════════════════════════════════════════════════════════
# Label mapping: All datasets → GoEmotions Full (28 base labels)
#
# GoEmotions Full 28 labels:
#   nothing, happy, angered, saddened, disgusted, surprised,
#   afraid, admiration, amusement, approval, caring, confused,
#   curious, desire, disappointed, disapproval, embarassed,
#   excited, grateful, love, nervous, optimistic, prideful,
#   realized, relieved, remorseful, grief, annoyed
# ══════════════════════════════════════════════════════════════

# TransMistral (MELD / MaSaC ERC / MaSaC EFR) → GoEmotions Full
TRANSMISTRAL_MAP = {
    "neutral":   "nothing",
    "joy":       "happy",
    "anger":     "angered",
    "sadness":   "saddened",
    "disgust":   "disgusted",
    "surprise":  "surprised",
    "fear":      "afraid",
    "contempt":  "disapproval",
}

# RVISA (SemEval aspect polarity) → GoEmotions Full
RVISA_MAP = {
    "positive":  "happy",
    "negative":  "disappointed",
    "neutral":   "nothing",
    "conflict":  "confused",
}

# GoEmotions Ekman → GoEmotions Full
GOEMO_EKMAN_MAP = {
    "happy":     "happy",
    "saddened":  "saddened",
    "nothing":   "nothing",
    "angered":   "angered",
    "disgusted": "disgusted",
    "surprised": "surprised",
    "scared":    "afraid",
}

# EmoEvent EN → GoEmotions Full
EMOEVENT_EN_MAP = {
    "happy":      "happy",
    "sad":        "saddened",
    "angry":      "angered",
    "disgusted":  "disgusted",
    "surprised":  "surprised",
    "scared":     "afraid",
    "nothing":    "nothing",
}

# EmoEvent ES (Spanish) → GoEmotions Full
EMOEVENT_ES_MAP = {
    "feliz":        "happy",
    "triste":       "saddened",
    "enojado":      "angered",
    "desagradado":  "disgusted",
    "sorprendido":  "surprised",
    "asustado":     "afraid",
    "nada":         "nothing",
}

# Mapping selector per dataset tag
DATASET_LABEL_MAP = {
    "transmistral_MELD":       TRANSMISTRAL_MAP,
    "transmistral_MaSaC_ERC":  TRANSMISTRAL_MAP,
    "transmistral_MaSaC_EFR":  TRANSMISTRAL_MAP,
    "rvisa_laptops":           RVISA_MAP,
    "rvisa_restaurants":       RVISA_MAP,
    "masive_GoEmo_Ekman":      GOEMO_EKMAN_MAP,
    "masive_GoEmo_Full":       None,   # Already GoEmotions Full — keep as-is
    "masive_EmoEvent_EN":      EMOEVENT_EN_MAP,
    "masive_EmoEvent_ES":      EMOEVENT_ES_MAP,
}


def normalize_label(raw_label: str, dataset_tag: str) -> str:
    """
    Map a raw label to GoEmotions Full format.
    
    For multi-label (e.g. RVISA with 'positive, negative'),
    each sub-label is mapped individually.
    GoEmotions Full labels are returned as-is.
    """
    label_map = DATASET_LABEL_MAP.get(dataset_tag)
    if label_map is None:
        # Already GoEmotions Full — just clean bracket format
        import ast
        try:
            parsed = ast.literal_eval(raw_label)
            if isinstance(parsed, list):
                return ", ".join(parsed)
        except:
            pass
        return raw_label.strip()

    # Clean the raw label
    import ast
    try:
        parsed = ast.literal_eval(raw_label)
        if isinstance(parsed, list):
            raw_label = ", ".join(str(x) for x in parsed)
    except:
        pass

    # Map each sub-label
    parts = [p.strip() for p in raw_label.split(",")]
    mapped = []
    for p in parts:
        p_lower = p.lower().strip()
        mapped_label = label_map.get(p_lower, p_lower)
        mapped.append(mapped_label)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for m in mapped:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    return ", ".join(unique)



def build_transmistral(limit: int) -> list[dict]:
    """Each utterance -> 1 sample. Context = other utterances in episode."""
    samples = []
    files = [
        ("MELD_test_efr.json",  "en", "transmistral_MELD"),
        ("MaSaC_test_erc.json", "mixed", "transmistral_MaSaC_ERC"),
        ("MaSaC_test_efr.json", "mixed", "transmistral_MaSaC_EFR"),
    ]
    for fname, lang, tag in files:
        path = DATA / "transmistral" / fname
        if not path.exists():
            print(f"  [SKIP] {fname}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            episodes = json.load(f)
        print(f"  {fname}: {len(episodes)} episodes")

        for ep in episodes:
            ep_name  = ep.get("episode", "unknown")
            utts     = ep.get("utterances", [])
            speakers = ep.get("speakers", [])
            emotions = ep.get("emotions", ep.get("labels", []))

            for i, text in enumerate(utts):
                if limit and len(samples) >= limit:
                    break
                speaker = speakers[i] if i < len(speakers) else "?"
                emotion = emotions[i] if i < len(emotions) else "unknown"

                # Build context = other utterances (not including self)
                ctx_lines = []
                for j, u in enumerate(utts):
                    if j == i:
                        continue
                    sp = speakers[j] if j < len(speakers) else "?"
                    ctx_lines.append(f"[{sp}]: {u}")
                context = "\n".join(ctx_lines)

                samples.append({
                    "id": f"{tag}_{ep_name}_U{i+1}",
                    "text": text,
                    "context": context,
                    "true_label": str(emotion),
                    "dataset": tag,
                })
            if limit and len(samples) >= limit:
                break
    return samples


def build_rvisa(limit: int) -> list[dict]:
    """Each XML sentence -> 1 sample. No context."""
    samples = []
    files = [
        ("laptops/Laptops_Test_Gold.xml", "rvisa_laptops"),
        ("restaurants/Restaurants_Test_Gold.xml", "rvisa_restaurants"),
    ]
    for rel, tag in files:
        path = DATA / "rvisa" / rel
        if not path.exists():
            print(f"  [SKIP] {rel}")
            continue
        tree = ET.parse(path)
        sents = tree.getroot().findall(".//sentence")
        print(f"  {rel}: {len(sents)} sentences")

        for sent in sents:
            if limit and len(samples) >= limit:
                break
            sid = sent.get("id", "?")
            txt = (sent.find("text").text or "") if sent.find("text") is not None else ""

            # True label = polarity only (comma-separated if multiple)
            at_el = sent.find("aspectTerms")
            if at_el is not None:
                aspects = at_el.findall("aspectTerm")
                polarities = [a.get("polarity", "unknown") for a in aspects]
                true_label = ", ".join(polarities)
            else:
                true_label = "neutral"

            samples.append({
                "id": f"{tag}_{sid}",
                "text": txt,
                "context": "",
                "true_label": true_label,
                "dataset": tag,
            })
    return samples


def build_masive(limit: int) -> list[dict]:
    """Each CSV row -> 1 sample. No context."""
    samples = []
    files = [
        ("goemo_ekman_test.csv", "masive_GoEmo_Ekman"),
        ("goemo_full_test.csv", "masive_GoEmo_Full"),
        ("emo_event_en_test.csv", "masive_EmoEvent_EN"),
        ("emo_event_es_test.csv", "masive_EmoEvent_ES"),
    ]
    for fname, tag in files:
        path = DATA / "masive" / fname
        if not path.exists():
            print(f"  [SKIP] {fname}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"  {fname}: {len(rows)} texts")

        for row in rows:
            if limit and len(samples) >= limit:
                break
            tid = row.get("id", f"r{len(samples)}")
            txt = row.get("text", "")
            lbl_raw = row.get("label_txt", row.get("label", "unknown"))
            # Clean "['saddened']" -> "saddened", "['happy', 'joy']" -> "happy, joy"
            lbl = str(lbl_raw).strip("[]'").replace("'", "").replace("\"" , "")

            samples.append({
                "id": f"{tag}_{tid}",
                "text": txt,
                "context": "",
                "true_label": str(lbl),
                "dataset": tag,
            })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20,
                    help="Max samples PER dataset (0=all). Default: 20")
    ap.add_argument("--output", default="data/eval_dataset.jsonl")
    args = ap.parse_args()

    all_samples = []
    print("=== TRANSMISTRAL ===")
    all_samples.extend(build_transmistral(args.limit))
    print("=== RVISA ===")
    all_samples.extend(build_rvisa(args.limit))
    print("=== MASIVE ===")
    all_samples.extend(build_masive(args.limit))

    # ── Normalize all labels to GoEmotions Full ──
    for s in all_samples:
        s["true_label"] = normalize_label(s["true_label"], s["dataset"])

    # ── Interleave: round-robin across sub-datasets ──
    # Group samples by sub-dataset
    from collections import defaultdict, Counter
    buckets = defaultdict(list)
    for s in all_samples:
        buckets[s["dataset"]].append(s)

    # Round-robin: pick 1 from each dataset in turn
    # When a dataset is exhausted, remove it (cycle shrinks 6→5→...→0)
    dataset_keys = list(buckets.keys())
    interleaved = []
    while dataset_keys:
        exhausted = []
        for ds in dataset_keys:
            if buckets[ds]:
                interleaved.append(buckets[ds].pop(0))
            else:
                exhausted.append(ds)
        for ds in exhausted:
            dataset_keys.remove(ds)

    # Assign numeric IDs after interleaving
    for i, s in enumerate(interleaved, start=1):
        s["id"] = i

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in interleaved:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(interleaved)} samples (interleaved)")
    print(f"Saved: {out}")

    # Print stats
    ds_counts = Counter(s["dataset"] for s in interleaved)
    for ds, cnt in ds_counts.items():
        print(f"  {ds}: {cnt} samples")


if __name__ == "__main__":
    main()
