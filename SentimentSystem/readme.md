# Text-Only Deep Reasoning Architecture (API-Based)
**TransMistral (Context Engine) + RVISA (Logic Core) + MASIVE (Vocabulary Head)**

This README is an implementation-oriented specification for a **text-only** emotion + psychological state analysis pipeline over:
- Facebook comment threads (tree-structured discussions)
- Chat logs (customer support, community chats, DMs)
- Forum threads (replies, quotes, nested discussions)

The pipeline is designed to solve three core problems end-to-end:

1) **Context** — handle long conversational history, speaker interactions, and identify **emotion flips** (where sentiment/emotion changes).  
2) **Logic** — infer *why* a user feels something (aspect → cause) and support implicit sentiment & sarcasm using **reasoning + verification**.  
3) **Granularity** — output fine-grained affective states (1,000+ style vocabulary) rather than only coarse emotion classes.

> **Key product promise:** not only “Angry”, but “**frustrated due to neglect**” backed by conversational evidence and verified reasoning.

---

## Table of Contents
1. [System Overview](#system-overview)  
2. [End-to-End Flow](#end-to-end-flow)  
3. [Data Contracts and Step Connectivity](#data-contracts-and-step-connectivity)  
4. [Step 0 Deep Dive — Ingestion + Preprocess + Normalization](#step-0-deep-dive--ingestion--preprocess--normalization)  
5. [Step 1 Deep Dive — TransMistral Contextual Parsing](#step-1-deep-dive--transmistral-contextual-parsing)  
6. [Step 2 Deep Dive — RVISA Reasoning Verification](#step-2-deep-dive--rvisa-reasoning-verification)  
7. [Step 3 Deep Dive — MASIVE Fine-Grained Labeling](#step-3-deep-dive--masive-fine-grained-labeling)  
8. [Orchestrator (API-Based) Design](#orchestrator-api-based-design)  
9. [Reliability: JSON Validation, Repair, Retries, Fallbacks](#reliability-json-validation-repair-retries-fallbacks)  
10. [Cost/Latency Controls: Anchor Gating, Windowing, Caching](#costlatency-controls-anchor-gating-windowing-caching)  
11. [Final Output Specification (What You Store/Return)](#final-output-specification-what-you-storereturn)  
12. [End-to-End Walkthrough Example](#end-to-end-walkthrough-example)  
13. [References](#references)  

---

## System Overview

### Components and Responsibilities
1) **TransMistral (Context Engine)**  
   - Reads the **whole conversation** (or summary+tail).  
   - Produces:
     - **coarse emotion timeline** (per utterance)
     - **emotion flips** (+ flip types)
     - **anchor utterances** (where deep reasoning is needed)
     - global **context_summary** (and optional embeddings)

2) **RVISA (Logic Core)**  
   - Runs only on selected anchors for cost/latency control.  
   - Two-stage:
     - **Generator:** proposes structured reasoning (aspect → cause → inferred attitude) + evidence hints  
     - **Verifier:** checks if reasoning is supported by text, reduces hallucinations, may correct

3) **MASIVE (Vocabulary Head)**  
   - Converts verified reasoning into **fine-grained affective state labels** (short phrases).  
   - Optionally normalizes labels to a known vocabulary for BI/CRM.

### Why the combination works
- **TransMistral** is best at global context: long history, speaker relations, flip localization.  
- **RVISA** is best at “why”: causal interpretation + verification for implicit/sarcastic text.  
- **MASIVE** is best at “what exactly”: high-resolution state naming for downstream actions.

---

## End-to-End Flow

```text
Raw thread/chat/log (text-only)
  │
  ├─ Step 0: Ingest + Preprocess + Normalize
  │    Output: PreprocessedConversation
  │
  ├─ Step 1: Contextual Parsing (TransMistral endpoint)
  │    Output: TransMistralOutput
  │           - coarse emotions + probabilities
  │           - emotion flips
  │           - anchor utterances + scores
  │           - context_summary (+ optional embeddings)
  │
  ├─ Step 2: Reasoning Verification (RVISA Generator endpoint → RVISA Verifier endpoint)
  │    Output: RVISAOutput (verified reasoning + evidence spans + confidence)
  │
  ├─ Step 3: Fine-Grained Labeling (MASIVE endpoint)
  │    Output: MASIVEOutput (fine label + alt labels + confidence + normalization)
  │
  └─ Final Assembly
       Output: FinalEmotionReport (conversation-level report with full timeline)
```

---

## Data Contracts and Step Connectivity

This section answers:
- **Input/Output** of each step
- **Where each input comes from**
- **How outputs become inputs of the next step**
- **Final output definition**

### Contract A — Raw ConversationObject (system input)
**Source:** ingestion from FB/chat/forum exports.

```json
{
  "conversation_id": "string",
  "source": "facebook|chat|forum",
  "language_hint": "vi|en|mixed|unknown",
  "utterances": [
    {
      "utt_id": "string",
      "speaker_id": "string",
      "timestamp": "ISO-8601 or int",
      "text_raw": "string",
      "reply_to_utt_id": "string|null",
      "meta": {}
    }
  ]
}
```

### Contract B — PreprocessedConversation (Step 0 output → Step 1 input)
**Produced by:** Step 0 in your Orchestrator.  
**Consumed by:** Step 1 TransMistral endpoint.

```json
{
  "conversation_id": "string",
  "utterances": [
    {
      "utt_id": "string",
      "speaker_id": "string",
      "timestamp": "…",
      "text_raw": "…",
      "text_clean": "…",
      "lang": "vi|en|mixed",
      "text_translated": "string|null",
      "reply_to_utt_id": "string|null"
    }
  ],
  "preprocess_meta": {
    "pivot_language": "en",
    "translation_provider": "…",
    "emoji_preserved": true,
    "pii_redaction": true
  }
}
```

### Contract C — TransMistralOutput (Step 1 output → Step 2 input)
**Produced by:** Step 1 TransMistral endpoint.  
**Consumed by:** Orchestrator to build RVISA jobs (anchors + windows).

```json
{
  "conversation_id": "string",
  "context_summary": "string",
  "context_vector": [0.01, -0.02, 0.03],
  "coarse_timeline": [
    {
      "utt_id": "string",
      "coarse_emotion": "neutral|joy|sadness|anger|fear|disgust|surprise|mixed|unknown",
      "prob": 0.0,
      "flip_flag": false,
      "flip_type": "pos->neg|neg->pos|calm->anger|...|none"
    }
  ],
  "anchors": [
    {
      "utt_id": "string",
      "anchor_score": 0.0,
      "anchor_reason": "flip-trigger|high-arousal|sarcasm-marker|low-confidence|..."
    }
  ],
  "flip_events": [
    {"from_utt_id":"string","to_utt_id":"string","flip_type":"string","trigger_utt_id":"string"}
  ]
}
```

### Contract D — RVISAOutput (Step 2 output → Step 3 input)
**Produced by:** Step 2 RVISA generator + verifier endpoints.  
**Consumed by:** Step 3 MASIVE endpoint.

```json
{
  "conversation_id": "string",
  "utt_id": "string",
  "aspect": "string",
  "cause": "string",
  "inferred_attitude": "positive|negative|neutral|mixed",
  "verified_rationale": "string",
  "evidence_spans": [
    {"utt_id": "string", "char_start": 0, "char_end": 10}
  ],
  "verdict": "pass|fail",
  "confidence": 0.0
}
```

### Contract E — MASIVEOutput (Step 3 output → Final Report)
**Produced by:** Step 3 MASIVE endpoint.  
**Consumed by:** Orchestrator final assembly.

```json
{
  "conversation_id": "string",
  "utt_id": "string",
  "fine_grained_label": "string",
  "alt_labels": ["string", "string"],
  "confidence": 0.0,
  "normalization": {
    "method": "exact|embedding_match|llm_rerank|cluster",
    "matched_vocab_id": "string|null",
    "cluster_id": "string|null"
  }
}
```

---

## Step 0 Deep Dive — Ingestion + Preprocess + Normalization

### Step 0 Purpose
Transform messy, heterogeneous sources into a consistent representation that preserves emotional signal and enables downstream reasoning.

### Step 0 Inputs
- **ConversationObject** (raw utterances + metadata)

### Step 0 Outputs
- **PreprocessedConversation** (clean + standardized utterances)
  - stable speaker IDs
  - cleaned text preserving emotion markers
  - language tags
  - optional translated view for code-mixed/multilingual inputs
  - optional PII redaction tokens

### Step 0 Detailed Operations (recommended order)
#### 0.1 Ingestion & source normalization
- Facebook/forum:
  - parse comment tree
  - preserve reply edges (`reply_to_utt_id`)
  - extract timestamps and authors
- Chat logs:
  - map agent/customer to speaker IDs
  - preserve channel/room IDs if needed

#### 0.2 Thread flattening (tree → linear timeline)
Goal: produce a **time-ordered utterance list** while preserving reply structure.
- Sort by timestamp when available
- If timestamp missing:
  - use platform order + reply depth order
- Keep `reply_to_utt_id` so models can understand “who replies to whom”

#### 0.3 Speaker canonicalization
- Build `speaker_id` mapping
- Optional: label roles (`customer`, `agent`, `moderator`) if known
- Ensure stable IDs across re-ingestion (important for caching)

#### 0.4 Text cleaning (do not destroy emotion signal)
Do:
- Replace URLs → `<url>`
- Replace mentions → `<user>`
- Normalize whitespace
- Keep: emoji, punctuation intensity, elongated words (“soooo”, “điiiiii”), repeated punctuation

Do NOT:
- Remove emojis entirely
- Strip “!!!” or “???”
- Aggressively normalize repeated characters if you can keep a feature version

#### 0.5 PII redaction (recommended in API-based external calls)
Before calling third-party model APIs:
- phone → `<pii_phone>`
- email → `<pii_email>`
- address → `<pii_address>`
- order IDs → `<pii_order>` (business-specific)

Keep a reversible mapping only if you are allowed to (compliance dependent).

#### 0.6 Language detection
- Per utterance (fast path)
- If utterance is long / mixed:
  - per sentence detection (optional)

Store:
- `lang = vi|en|mixed|unknown`

#### 0.7 Translation view (optional but recommended)
If `lang == mixed` or you want consistent pivot:
- translate to pivot language (often EN)
- store in `text_translated`
- cache translations by hash of `text_clean`

**Output of Step 0 becomes the input of Step 1** directly.

### Step 0 Failure modes & mitigations
- Missing timestamps:
  - approximate order + store `ordering_confidence`
- Broken HTML/entities:
  - decode safely; keep a raw backup
- Excessive length:
  - chunking policy for Step 1 (handled there)

---

## Step 1 Deep Dive — TransMistral Contextual Parsing

### Step 1 Purpose
Given the **whole conversation context**, produce:
- **Coarse emotion per utterance**
- **Emotion flips**
- **Anchor utterances**
- **Global context representation** (summary + optional vector)

Step 1 is a *context engine*, not a deep reasoning engine. Its main job is to locate where deep reasoning is needed.

### Step 1 Inputs (from Step 0)
**TransMistralInput** is constructed entirely from **PreprocessedConversation**:
- utterances with speaker/time ordering
- text_clean + optional text_translated
- reply_to links

Recommended input fields:
- `conversation_id`
- `utterances[]: {utt_id, speaker_id, timestamp, text_clean, text_translated?, reply_to_utt_id?}`
- `settings: {pivot_language, max_context_tokens, context_strategy, return_vector}`

### Step 1 Outputs
**TransMistralOutput**:
- `context_summary`
- optional `context_vector`
- `coarse_timeline` (coarse emotion + prob per utterance)
- flip representation (either `flip_flag` per utterance and/or `flip_events[]`)
- `anchors[]` (utt_id + anchor_score + anchor_reason)

### Step 1 Conversation serialization (how you feed the model)
A stable, explicit format improves consistency.

**Recommended serialization**
- one utterance per line:
  - `[utt_id=U12][t=...][S=S2][reply_to=U9] raw:"..." | translated:"..."`
- If no translation, omit translated.
- Keep ordering fixed.

### Step 1 Long-context strategies (choose based on size & cost)
#### Strategy A — Full conversation
Use when total tokens fit comfortably under provider limits.

#### Strategy B — Sliding tail
Use only last N utterances (e.g., last 120 turns).  
Pros: cheaper, good for recent dynamics.  
Cons: may miss early context that explains later flips.

#### Strategy C — Summary + Tail (recommended default)
1) Create a cheap summary of older turns: `head_summary`
2) Provide latest `tail_window` of N utterances
3) TransMistral sees:
   - head_summary: what happened before
   - tail_window: what’s happening now

#### Strategy D — Two-pass refinement (best quality)
- Pass 1: detect flip zones + candidate anchors fast
- Pass 2: zoom around flip zones to refine anchor reasons and flip triggers

### Step 1 Prompt template (JSON-only output)
```text
SYSTEM:
You are a conversation context engine. Return ONLY valid JSON matching the schema.

USER:
Tasks:
1) Assign a coarse emotion to each utterance.
2) Detect emotion flips (where the trajectory changes and which utterance triggers it).
3) Select anchor utterances requiring deeper reasoning.

Schema:
{
  "context_summary": "string",
  "coarse_timeline": [
    {"utt_id":"...","coarse_emotion":"neutral|joy|sadness|anger|fear|disgust|surprise|mixed|unknown",
     "prob":0.0,"flip_flag":false,"flip_type":"none"}
  ],
  "anchors": [{"utt_id":"...","anchor_score":0.0,"anchor_reason":"..."}],
  "flip_events": [{"from_utt_id":"...","to_utt_id":"...","flip_type":"...","trigger_utt_id":"..."}]
}

Conversation:
[utt_id=U1][t=...][S=S1] raw:"..." | translated:"..."
...
```

### Step 1 Anchor selection logic (what should be an anchor)
Anchors should represent utterances that **need deeper logic**, such as:
- flip triggers (emotion changed here)
- high-arousal or high-risk emotions (anger/disgust/intense sadness)
- low-confidence coarse predictions (uncertain)
- sarcasm markers (surface cues + context mismatch)
- domain escalation patterns (refund request, “I’m leaving”, threats, abuse)

### Step 1 Post-processing (Orchestrator responsibilities)
1) Parse JSON strictly; validate schema.
2) Normalize coarse emotion to enum.
3) Optional: recompute / adjust anchor_score deterministically (recommended).
4) Final anchor gating:
   - keep `anchor_score >= T_anchor`  
   - cap to top `max_anchors`

### Step 1 → Step 2 connectivity (explicit)
- `anchors[]` becomes the **RVISA job list**.
- `context_summary` becomes the **global grounding** for RVISA prompts.
- `flip_events` determines **window expansion** rules for RVISA.

---

## Step 2 Deep Dive — RVISA Reasoning Verification

### Step 2 Purpose
For each anchor utterance, infer *why* the emotion exists and verify that reasoning against textual evidence.

Step 2 produces:
- **aspect** (what is being evaluated)
- **cause** (why user feels that way)
- **inferred_attitude** (pos/neg/neutral/mixed)
- **verified_rationale** (explanation grounded in context)
- **evidence_spans** (where in text support exists)
- **verdict + confidence** (pass/fail and reliability)

### Step 2 Inputs (where they come from)
Step 2 inputs are built by the Orchestrator from two sources:

1) **From Step 0 (PreprocessedConversation):**
- raw/clean/translated utterances needed to build context windows

2) **From Step 1 (TransMistralOutput):**
- anchor list (`anchors[]`)
- context summary (`context_summary`)
- flip events (to decide window sizes)

**Per-anchor RVISAInput**
- `anchor_utt_id`, `anchor_text`
- `window[]` = utterances before/after anchor (speaker-aware)
- `context_summary` = global memory from Step 1
- optional: Step 1 coarse emotion for anchor

### Step 2 Window building (critical for reasoning quality)
Recommended default:
- `k_before = 6`
- `k_after = 2`

If anchor is a flip trigger (or near flip zone):
- expand to `k_before = 12`, `k_after = 4`

If thread is extremely long:
- optionally include short summaries of earlier segments as extra context

**Window must preserve:**
- chronological ordering
- speaker IDs
- reply_to structure if helpful (optional field)

### Step 2A — RVISA Generator (hypothesis creation)
The generator proposes a structured reasoning hypothesis.
**Output must be machine-readable** (JSON) even if the model “thinks” internally.

#### Generator prompt template (JSON-only)
```text
SYSTEM:
You are RVISA Generator. Think internally, but output ONLY final JSON.

USER:
Given a context window and an anchor utterance, infer implicit sentiment using 3-step reasoning:
(1) Aspect identification
(2) Cause analysis
(3) Attitude inference

Return JSON:
{
  "aspect": "...",
  "cause": "...",
  "inferred_attitude": "positive|negative|neutral|mixed",
  "rationale": "...",
  "evidence": [{"utt_id":"...","quote":"..."}]
}

Global context summary: "..."
Window:
[U10][S1] "..."
[U11][S2] "..."
Anchor:
[U12][S2] "..."
```

#### Generator output interpretation
- The generator may still hallucinate; do not trust it blindly.
- Treat generator output as a *candidate* to be verified.

### Step 2B — RVISA Verifier (evidence checking & correction)
The verifier checks whether generator reasoning is supported by the window.

#### Verifier prompt template (JSON-only)
```text
SYSTEM:
You are RVISA Verifier. Verify whether the generator output is supported by evidence in the window.
Output ONLY JSON.

USER:
Return JSON:
{
  "verdict": "pass|fail",
  "confidence": 0.0,
  "corrected": {
    "aspect": "...",
    "cause": "...",
    "inferred_attitude": "...",
    "verified_rationale": "...",
    "evidence_spans": [{"utt_id":"...","char_start":0,"char_end":10}]
  }
}

Window: ...
Generator output: ...
```

### Pass/Fail rules (practical and enforceable)
**PASS if all are true:**
- aspect/cause are explicitly or strongly implied in window
- rationale does not invent new events/entities
- evidence spans point to real text content
- speaker references are consistent

**FAIL if any are true:**
- rationale relies on unsupported assumptions
- contradicts text in window
- wrong speaker attribution
- sarcasm is “interpreted” without contextual evidence

### Step 2 Outputs (RVISAOutput)
Your Orchestrator should store:
- `verdict`, `confidence`
- if PASS:
  - `verified_rationale` + evidence spans + aspect/cause/attitude
- if FAIL:
  - keep minimal info and mark as unverified (or drop Step 2 results)

### Step 2 → Step 3 connectivity (explicit)
- Step 3 consumes only **verified** reasoning:
  - `verified_rationale`, `aspect`, `cause`, `inferred_attitude`
- If Step 2 fails:
  - Step 3 is usually skipped (or run in fallback mode with lower confidence)

### Step 2 Failure modes & mitigations
- Window too small → misses causal evidence:
  - expand window around anchors, especially near flips
- Confusing reply structure:
  - include reply_to edges or quote markers in serialization
- Cost explosion due to too many anchors:
  - tighten gating and max_anchors

---

## Step 3 Deep Dive — MASIVE Fine-Grained Labeling

### Step 3 Purpose
Transform verified reasoning into a **fine-grained affective state** label suitable for:
- UX triggers (escalate, churn prevention, apology templates)
- BI dashboards (top emotional states by product area)
- CRM annotations (agent guidance)

Step 3 outputs:
- fine label (short phrase)
- alternative labels (top-k)
- confidence
- normalization mapping (optional but recommended)

### Step 3 Inputs (where they come from)
Step 3 inputs come directly from **Step 2 RVISAOutput** (PASS only):
- `verified_rationale`
- `aspect`
- `cause`
- `inferred_attitude`
- `pivot_language` preference (EN/VI)

### Step 3 Output (MASIVEOutput)
- `fine_grained_label`
- `alt_labels[]`
- `confidence`
- `normalization` metadata

### Step 3 Prompt template (open-ended MASIVE style)
```text
SYSTEM:
You are a MASIVE-style open-ended affective state identifier. Return ONLY JSON.

USER:
Given the verified rationale, output the most precise affective-state label as a short phrase.

Return JSON:
{
  "fine_grained_label": "...",
  "alt_labels": ["...", "..."],
  "confidence": 0.0
}

Verified rationale: "..."
Aspect: "..."
Cause: "..."
Attitude: "..."
```

### Label constraints (recommended)
To keep labels usable:
- 2–6 words (English) or 2–8 words (Vietnamese)
- Avoid overly generic labels (“negative”, “sad”) unless necessary
- Prefer psychologically meaningful phrases:
  - “betrayed”, “dismissive resentment”, “frustrated due to neglect”, “anxious uncertainty”

### Step 3 Normalization (highly recommended)
Open-ended labels are hard to aggregate. Choose one approach:

#### Option A — Vocabulary-constrained (best for BI)
- Maintain a vocabulary list (e.g., 1,000+ states)
- Normalize outputs via:
  1) exact match
  2) embedding nearest neighbor
  3) optional LLM re-rank among top N candidates

Output should include:
- `matched_vocab_id`

#### Option B — Clustering (if you don’t have vocab)
- Embed labels (or label+reasoning text)
- Cluster into 50–200 buckets
- Store `cluster_id` as normalized key

### Step 3 → Final output connectivity (explicit)
- Step 3 label attaches to the anchor’s timeline entry.
- Optional: propagate label influence to a span of turns (until next flip) for UI summaries.

### Step 3 Failure modes & mitigations
- Label drift (“creative phrases”):
  - enforce vocabulary constraints or embedding-match normalization
- Low confidence when rationale is vague:
  - store label with low confidence; prefer coarse results in UI

---

## Orchestrator (API-Based) Design

### Why an Orchestrator is mandatory
API-based pipelines must handle:
- contract enforcement (schemas)
- retries and JSON repair
- cost/latency gating
- aggregation and storage
- caching across steps

### Suggested architecture
- `orchestrator-api` (your service)
- optional `translation-service`
- model endpoints:
  - `transmistral-context-engine`
  - `rvisa-generator`
  - `rvisa-verifier`
  - `masive-mt5`
- storage:
  - document DB (final report JSON)
  - optional vector DB (embeddings)

### Orchestrator endpoints
#### Sync
- `POST /v1/analyze` → returns `FinalEmotionReport`

#### Async (recommended for long logs)
- `POST /v1/jobs` → returns `job_id`
- `GET /v1/jobs/{job_id}` → returns status and result
- optional webhook callback

### Orchestrator step-by-step algorithm
1) Receive raw `ConversationObject`.
2) Run Step 0 preprocess:
   - flatten thread, clean text, detect language, translate if configured, redact PII.
3) Call Step 1 TransMistral endpoint:
   - get coarse_timeline + flips + anchors + context_summary.
4) Anchor gating:
   - select anchors by threshold + max_anchors + risk rules.
5) For each anchor:
   - build context window from Step 0 data (expand near flips).
   - call RVISA generator, then verifier.
   - if PASS:
     - call MASIVE for fine label
     - normalize label
   - if FAIL:
     - fallback to Step 1 coarse output for that anchor
6) Assemble final report:
   - merge Step 1 results for all turns
   - attach Step 2 + Step 3 results on anchors
   - compute aggregates (dominant states, key anchors)
7) Store final report and return it (sync) or mark job done (async).

---

## Reliability: JSON Validation, Repair, Retries, Fallbacks

### JSON validation (per step)
For Step 1/2/3 model outputs:
1) Strict parse as JSON
2) Validate schema (JSONSchema or Pydantic)
3) If invalid:
   - retry once with stricter instruction (“ONLY JSON, match schema”)
4) If still invalid:
   - run JSON repair step (deterministic fixer or a dedicated repair model)
5) If still invalid:
   - fallback:
     - Step 1: attempt minimal coarse-only output
     - Step 2: mark anchor unverified
     - Step 3: omit fine label

### Retries and backoff
- Retry transient HTTP errors
- Exponential backoff with jitter
- Circuit breaker per provider to prevent cascading failures

### Fallback policy (recommended)
- If Step 1 fails: return preprocessing-only metadata + error
- If Step 2 fails for an anchor:
  - keep Step 1 coarse emotion + mark `unverified`
- If Step 3 fails:
  - keep Step 2 verified rationale but omit fine label (or label="unknown")

---

## Cost/Latency Controls: Anchor Gating, Windowing, Caching

### Anchor gating (primary control knob)
Run Step 2/3 only on anchors.

Recommended defaults:
- `T_anchor = 0.65`
- `max_anchors = 20`
- `low_conf_prob = 0.55`
- Always include flip triggers even if score slightly below threshold

### Windowing (control token size)
- Default `k_before=6`, `k_after=2`
- Expand near flips: `k_before=12`, `k_after=4`
- For extremely long threads, add short segment summaries instead of raw turns

### Caching (major cost reducer)
Cache by step and model version:

- Step 1: `hash(conversation_clean + model_version + pivot_lang)`
- Step 2: `hash(anchor_text + window_text + model_version)`
- Step 3: `hash(verified_rationale + model_version)`

### Observability
- Correlation ID per request/job
- Log per step:
  - latency, tokens, retries, schema failures, anchor counts
- Debug mode (secure):
  - store raw model outputs, repaired JSON, and windows

---

## Final Output Specification (What You Store/Return)

### FinalEmotionReport
This is the **final output** of the entire pipeline.

```json
{
  "conversation_id": "string",
  "context_summary": "string",
  "timeline": [
    {
      "utt_id": "string",
      "speaker_id": "string",
      "timestamp": "…",
      "text_preview": "string",
      "coarse": {
        "emotion": "anger",
        "prob": 0.87,
        "flip_flag": false,
        "flip_type": "none"
      },
      "anchor": {
        "is_anchor": true,
        "score": 0.81,
        "reason": "flip-trigger"
      },
      "rvisa": {
        "verdict": "pass",
        "confidence": 0.76,
        "aspect": "string",
        "cause": "string",
        "verified_rationale": "string",
        "evidence_spans": [{"utt_id":"string","char_start":0,"char_end":10}]
      },
      "masive": {
        "label": "frustrated due to neglect",
        "alt": ["ignored", "disrespected"],
        "confidence": 0.68,
        "normalization": {
          "method": "embedding_match",
          "matched_vocab_id": "M_0421",
          "cluster_id": null
        }
      }
    }
  ],
  "key_anchors": ["utt_id", "utt_id"],
  "dominant_states": ["string", "string"],
  "meta": {
    "model_versions": {"step0":"...","step1":"...","step2":"...","step3":"..."},
    "options": {"max_anchors":20,"window_before":6,"window_after":2}
  }
}
```

### What the final output represents
- **coarse** exists for every utterance (fast, broad coverage).
- **rvisa + masive** exist mainly on anchors (deep, precise).
- **dominant_states** helps summarize conversation-level emotional state.
- Evidence spans provide auditability for reasoning.

---

## End-to-End Walkthrough Example

Imagine a 30-turn customer support chat:
- The user starts neutral, becomes annoyed after repeated delays, then angry after being ignored.

**Step 0**
- Ingest chat export → standardized utterances
- Detect language mixed VI/EN
- Create translated view (pivot EN)
- Redact PII (order IDs, phone)

**Step 1 (TransMistral)**
- Output coarse timeline (neutral → annoyance/anger)
- Detect flip event at U18 (calm → anger)
- Anchors include:
  - U18 (flip-trigger)
  - U22 (sarcasm marker: “Wow great service…”)

**Anchor gating**
- Keep top 10 anchors, always keep flip triggers

**Step 2 (RVISA)**
- Build window around U18 (expand due to flip)
- Generator proposes:
  - aspect: responsiveness
  - cause: ignored for 48 hours
- Verifier checks evidence spans (mentions “2 days no response”) → PASS

**Step 3 (MASIVE)**
- Input: verified rationale
- Output: “frustrated due to neglect” + alternatives
- Normalize to nearest vocab label

**Final report**
- Full timeline with coarse emotions everywhere
- Anchors show verified “why” + fine labels + evidence
- Dominant state: “frustrated due to neglect”

---

## References
- RVISA paper: https://ira.lib.polyu.edu.hk/bitstream/10397/112805/1/Lai_RVISA_Reasoning_Verification.pdf  
- RVISA landing page: https://repository.eduhk.hk/en/publications/rvisa-reasoning-and-verification-for-implicit-sentiment-analysis/  
- MASIVE (ACL Anthology): https://aclanthology.org/2024.emnlp-main.1139/  
- MASIVE (PDF): https://aclanthology.org/2024.emnlp-main.1139.pdf  
- MASIVE (arXiv): https://arxiv.org/abs/2407.12196  
- MASIVE code: https://github.com/NickDeas/MASIVE  
- TransMistral (SemEval paper): https://aclanthology.org/2024.semeval-1.46.pdf  
- TransMistral (ResearchGate mirror): https://www.researchgate.net/publication/382633282_TransMistral_at_SemEval-2024_Task_10_Using_Mistral_7B_for_Emotion_Discovery_and_Reasoning_its_Flip_in_Conversation
