# A Text-Only Deep Reasoning Architecture for Fine-Grained Conversational Sentiment Analysis

## Abstract

We propose a modular, API-based pipeline for fine-grained sentiment and affective state analysis over text-only conversational data, including social media comment threads, customer support chat logs, and forum discussions. Our architecture combines three complementary components: **TransMistral**, a context engine for coarse emotion classification and emotion flip detection across conversational history; **RVISA** (Reasoning and Verification for Implicit Sentiment Analysis), a two-stage generator–verifier framework for causal reasoning and evidence grounding; and **MASIVE** (Multilingual Affect and Sentiment Identification with Versatile Expressions), a vocabulary head that produces fine-grained affective state labels from verified reasoning. The system is orchestrated through a sequential four-step pipeline with anchor-gating for cost control, JSON schema enforcement for reliability, and structured data contracts between stages. Rather than outputting only coarse emotion categories, the pipeline delivers psychologically meaningful labels (e.g., *"frustrated due to neglect"*) backed by verified conversational evidence.

**Keywords:** sentiment analysis, emotion detection, conversational AI, implicit sentiment, reasoning verification, fine-grained affect labeling, large language models

---

## I. Introduction

Sentiment analysis in conversational settings presents challenges beyond those encountered in document-level or sentence-level tasks. Conversations involve multiple speakers, evolving emotional trajectories, implicit attitudes, sarcasm, and complex pragmatic phenomena that require both contextual understanding and causal reasoning.

Existing approaches typically address these challenges in isolation: contextual models capture speaker dynamics but lack causal depth; reasoning-based approaches provide causal interpretation but struggle with long-range context; and fine-grained labeling systems offer rich vocabularies but do not ground their outputs in conversational evidence.

We propose an integrated architecture that addresses three core problems end-to-end:

1. **Context** — Handling long conversational history, speaker interactions, and identifying emotion flips where sentiment trajectories change.
2. **Logic** — Inferring *why* a user feels something (aspect → cause) and supporting implicit sentiment and sarcasm detection through structured reasoning with evidence verification.
3. **Granularity** — Producing fine-grained affective state labels (1,000+ potential expressions) rather than only coarse emotion classes.

Our key contribution is the combination of three research-grounded components — TransMistral, RVISA, and MASIVE — into a single API-based pipeline with formal data contracts, cost control mechanisms, and reliability guarantees.

---

## II. Related Work

### A. Contextual Emotion Recognition in Conversations

TransMistral (Sharma et al., 2024) addresses emotion recognition and flip detection in multi-turn conversations by leveraging the Mistral-7B architecture. The approach was evaluated at SemEval-2024 Task 10, demonstrating competitive performance on Emotion Discovery and Reasoning its Flip in Conversation (EDiReF). We adapt this approach as our context engine (Step 1), using the Mistral API to produce coarse emotion timelines and identify anchor utterances requiring deeper analysis.

### B. Implicit Sentiment Analysis via Reasoning

RVISA (Lai et al., 2024) proposes a two-stage reasoning–verification framework for implicit sentiment analysis. The generator stage produces structured reasoning hypotheses (aspect identification → cause analysis → attitude inference), while the verifier stage checks whether the proposed reasoning is grounded in textual evidence. This approach significantly reduces hallucination compared to single-pass reasoning. We implement RVISA as our logic core (Step 2) in a generator–verifier configuration.

### C. Fine-Grained Affective State Labeling

MASIVE (Deas et al., 2024) introduces a multilingual framework for open-ended affective state identification that goes beyond fixed emotion taxonomies. Rather than constraining outputs to a small set of categories, MASIVE generates psychologically meaningful short phrases (e.g., *"dismissive resentment"*, *"anxious uncertainty"*). We adopt this as our vocabulary head (Step 3), converting verified reasoning into actionable labels suitable for CRM, UX triggers, and business intelligence dashboards.

---

## III. Proposed Method

### A. System Architecture Overview

The proposed system consists of four sequential processing stages connected by formal data contracts, managed by a central orchestrator. The pipeline enforces typed inter-stage contracts implemented as Pydantic models, enabling automatic schema validation at every stage boundary. Table I summarizes the data flow.

| Contract | Name | Producer | Consumer |
|---|---|---|---|
| A | ConversationObject | External ingestion | Step 0 |
| B | PreprocessedConversation | Step 0 | Step 1 |
| C | TransMistralOutput | Step 1 | Orchestrator → Step 2 |
| D | RVISAOutput | Step 2 | Step 3 |
| E | MASIVEOutput | Step 3 | Final Assembly |
| Final | FinalEmotionReport | Orchestrator | Consumer application |

*TABLE I: Data contracts between pipeline stages.*

### B. Step 0: Ingestion, Preprocessing, and Normalization

The preprocessing stage transforms heterogeneous raw conversational data into a standardized representation that preserves emotional signals while enabling downstream API-based processing.

The system accepts a `ConversationObject` containing raw utterances with speaker identifiers, timestamps, and reply-to relationships. Supported sources include Facebook comment exports, customer support chat logs, and forum thread dumps.

The preprocessing pipeline applies the following operations in order:

1. **Thread Flattening**: Utterances are sorted into a time-ordered linear list. For tree-structured discussions (Facebook, forums), the system preserves `reply_to_utt_id` edges while producing a chronologically ordered sequence.

2. **Speaker Canonicalization**: Speaker identifiers are mapped to stable canonical IDs (S1, S2, ...) to ensure consistency across re-ingestion and to support caching.

3. **Text Cleaning**: URLs are replaced with `<url>` tokens; @mentions are replaced with `<user>` tokens; whitespace is normalized. Critically, emotion-bearing signals — emojis, punctuation intensity ("!!!", "???"), and elongated words ("soooo") — are *preserved*, as they carry significant affective information.

4. **PII Redaction**: Before any external API call, personally identifiable information is redacted: phone numbers → `<pii_phone>`, email addresses → `<pii_email>`.

5. **Language Detection**: Per-utterance language is detected using `langdetect` with a Vietnamese Unicode heuristic fallback. The output tag is one of `{vi, en, mixed, unknown}`.

6. **Translation View** (optional): For multilingual or code-mixed inputs, utterances are translated to a pivot language (default: English) using `deep-translator`. Translations are cached by hash of `text_clean`.

The output is a `PreprocessedConversation` (Contract B) containing cleaned utterances with language tags, optional translations, and preprocessing metadata.

### C. Step 1: TransMistral — Contextual Parsing

TransMistral serves as the *context engine*. Given the full preprocessed conversation, it produces:

- A **coarse emotion timeline**: one emotion label per utterance from the set {neutral, joy, sadness, anger, fear, disgust, surprise, mixed, unknown}, with associated probabilities.
- **Emotion flip events**: transitions in emotional trajectory (e.g., calm→anger), including the trigger utterance.
- **Anchor utterances**: utterances flagged for deeper reasoning based on flip-trigger presence, high arousal, sarcasm markers, low prediction confidence, or domain escalation patterns.
- A **context summary**: a natural language summary of the conversational dynamics.

Conversations are serialized into a stable, explicit format where each utterance occupies one line with structured metadata tags (utterance ID, timestamp, speaker ID, reply-to reference) followed by the raw and optionally translated text.

For conversations exceeding the model's context window, we implement a configurable strategy selection as shown in Table II.

| Strategy | Description | Trade-off |
|---|---|---|
| Full conversation | All utterances in context | Best quality, highest cost |
| Sliding tail | Last N utterances only | Low cost, may miss early context |
| Summary + Tail (default) | Summary of older turns + tail window | Balanced cost and quality |
| Two-pass refinement | Pass 1: detect flip zones; Pass 2: refine | Best quality, double cost |

*TABLE II: Long-context handling strategies for Step 1.*

The system prompt instructs the model to return only valid JSON conforming to a predefined schema. The user prompt specifies three tasks: (1) coarse emotion assignment, (2) flip detection, and (3) anchor selection. On JSON parse failure, the system retries with a stricter prompt prefix.

Anchors represent utterances that require deeper logical analysis. Selection criteria include: flip triggers (where emotion trajectory changes), high-arousal emotions (anger, disgust, intense sadness), low-confidence predictions, sarcasm markers (surface-level cues with context mismatch), and domain escalation patterns (refund requests, threats, abuse).

### D. Anchor Gating

Between Steps 1 and 2, an anchor gating mechanism controls cost and latency by filtering the anchor set:

- Anchors with `anchor_score ≥ T_anchor` (default: 0.65) are retained
- Flip triggers are always included regardless of score
- The total is capped at `max_anchors` (default: 20), ranked by score

This ensures that Steps 2 and 3 — which involve multiple LLM calls per anchor — operate only on the most informative utterances.

### E. Step 2: RVISA — Reasoning and Verification

For each gated anchor, RVISA infers *why* the emotion exists and verifies that the reasoning is grounded in textual evidence. This two-stage approach reduces hallucination compared to single-pass reasoning.

The orchestrator builds a context window around each anchor from the preprocessed conversation. The default window includes k=6 utterances before and k=2 after the anchor. Near flip zones, the window expands to k=12 before and k=4 after. The window preserves chronological ordering, speaker IDs, and reply-to structure.

**Stage 2A — Generator.** The generator receives the context window, the anchor utterance, and the global context summary from Step 1. It produces a structured reasoning hypothesis containing: aspect (what is being evaluated), cause (why the user feels that way), inferred attitude (positive/negative/neutral/mixed), rationale, and evidence hints. The generator output is treated as a *candidate hypothesis* — not a trusted result.

**Stage 2B — Verifier.** The verifier independently evaluates whether the generator's reasoning is supported by evidence in the conversation window.

Pass criteria (all must hold): aspect and cause are explicitly stated or strongly implied in the window; rationale does not fabricate events or entities; evidence spans reference actual text content; speaker attributions are correct.

Fail criteria (any triggers failure): rationale relies on unsupported assumptions; contradicts text in the window; incorrect speaker attribution; sarcasm interpretation without contextual grounding.

The verifier may also produce a *corrected* version of the reasoning with refined evidence spans and confidence scores. The output is an `RVISAOutput` (Contract D) containing: verdict (pass/fail), confidence score, verified aspect, cause, inferred attitude, rationale, and character-level evidence spans. Only results with verdict=PASS proceed to Step 3.

### F. Step 3: MASIVE — Fine-Grained Affective State Labeling

MASIVE converts verified reasoning into fine-grained affective state labels suitable for downstream applications: UX triggers, BI dashboards, and CRM annotations.

The model receives the verified rationale, aspect, cause, and inferred attitude from Step 2. It generates a primary label (a psychologically meaningful phrase of 2–6 words in English or 2–8 words in Vietnamese), top-2 alternative labels, and a confidence score. Labels are constrained to avoid generic terms (e.g., "negative", "sad") in favor of psychologically meaningful expressions such as *"frustrated due to neglect"*, *"betrayed trust"*, or *"anxious uncertainty"*.

To support aggregation and downstream analytics, labels are normalized using one of four methods as shown in Table III.

| Method | Description | Suitable for |
|---|---|---|
| Exact match | Direct lookup in a curated vocabulary | Established taxonomies |
| Embedding match | Nearest-neighbor in embedding space | General-purpose |
| LLM rerank | LLM ranks top-N candidates from vocabulary | Highest accuracy |
| Clustering | Labels embedded and clustered into 50–200 buckets | When no vocabulary exists |

*TABLE III: Label normalization methods for Step 3.*

### G. Final Assembly

The orchestrator merges outputs from all steps into a `FinalEmotionReport`:

- **Full coarse timeline** covering every utterance (from Step 1)
- **Deep analysis results** on anchor utterances (from Steps 2–3)
- **Conversation-level aggregates**: dominant affective states, key anchors
- **Metadata**: model versions, pipeline configuration, processing options

---

## IV. Implementation

### A. System Configuration

The system is implemented in Python and communicates with the Mistral AI API (`mistral-small-latest`) for all LLM inference. The LLM interface layer manages multiple API keys with round-robin rotation, cooldown-based availability tracking, capacity tier monitoring, and infinite-retry mode with exponential backoff.

### B. Reliability Mechanisms

Each LLM call output undergoes a multi-stage validation pipeline:

1. Strict JSON parsing
2. Pydantic schema validation
3. On failure: retry with a stricter system prompt
4. On persistent failure: heuristic JSON repair (strip markdown fences, extract `{...}` blocks)
5. On exhaustion: step-specific fallback

The fallback policy is defined as follows: if Step 1 fails, only preprocessing metadata is returned; if Step 2 fails for an anchor, the Step 1 coarse emotion is retained and marked as *unverified*; if Step 3 fails, the verified rationale is retained with label set to *"unknown"*.

The system implements exponential backoff with jitter for transient HTTP errors. API keys are managed through round-robin rotation with per-key cooldown periods and rate-limit–aware circuit breakers to prevent cascading failures.

### C. Cost Control Mechanisms

Four mechanisms control pipeline cost and latency: (1) anchor gating limits Step 2/3 calls to high-value utterances; (2) context windowing bounds per-call token usage; (3) per-step caching avoids redundant calls via content-hash keys; (4) observability through correlation IDs and per-step latency/token/retry logging.

---

## V. Experimental Evaluation

We illustrate the pipeline using a 7-turn Vietnamese customer support conversation where a customer progresses from neutral inquiry through frustration to resigned disappointment.

**Step 0**: The chat export is ingested and normalized. Language is detected as Vietnamese. PII patterns are redacted. Emotionally significant punctuation ("???", "...") is preserved.

**Step 1 (TransMistral)**: The context engine produces a coarse timeline: U1 (neutral) → U2 (neutral) → U3 (anger, p=0.87, flip) → U4 (neutral) → U5 (anger, p=0.91) → U6 (neutral) → U7 (sadness, p=0.72). A flip event is detected at U2→U3 (calm→anger). Three anchors are selected: U3 (flip-trigger), U5 (high-arousal), U7 (mixed-signal).

**Anchor Gating**: All three anchors exceed the threshold (T=0.65) and are retained.

**Step 2 (RVISA)** for anchor U5: The generator identifies aspect="service reliability", cause="repeated promises to check without resolution". The verifier confirms evidence in U3 ("waited 2 weeks") and U5 ("every time you just say you'll check") — verdict: PASS, confidence=0.82.

**Step 3 (MASIVE)** for anchor U5: The verified rationale yields the label *"frustrated due to broken promises"* with alternatives *"exasperated disappointment"* and *"exhausted patience"* (confidence=0.78).

**Final Report**: A complete timeline with coarse emotions for all 7 utterances, deep analysis on 3 anchors, and dominant conversational state: *"frustrated due to broken promises"*.

---

## VI. Conclusion

We presented a modular, API-based architecture for fine-grained conversational sentiment analysis that combines three complementary components: TransMistral for contextual parsing and emotion flip detection, RVISA for structured causal reasoning with evidence verification, and MASIVE for psychologically meaningful affective state labeling. The four-step pipeline — preprocessing, contextual parsing, reasoning verification, and fine-grained labeling — is connected by formal Pydantic data contracts and managed by a central orchestrator with anchor gating for cost control and multi-level fallback for reliability.

The architecture addresses three fundamental limitations of existing conversational sentiment systems: (1) the inability to jointly model long-range context and causal reasoning, (2) the lack of evidence grounding for inferred emotions, and (3) the restriction to coarse emotion taxonomies. By combining context-aware analysis with verified reasoning and open-ended labeling, the system produces outputs such as *"frustrated due to neglect"* that are both interpretable and actionable for downstream applications including CRM, UX design, and business intelligence.

Future work includes expanding to multimodal inputs (voice, images), integrating real-time streaming analysis, and conducting large-scale evaluation across diverse conversational domains and languages.

---

## References

[1] H. Lai et al., "RVISA: Reasoning and Verification for Implicit Sentiment Analysis," in *Proc. EMNLP*, 2024. [Online]. Available: https://ira.lib.polyu.edu.hk/bitstream/10397/112805/1/Lai_RVISA_Reasoning_Verification.pdf

[2] N. Deas et al., "MASIVE: Open-Ended Affective State Identification in English and Spanish," in *Proc. EMNLP*, 2024, pp. 1139. [Online]. Available: https://aclanthology.org/2024.emnlp-main.1139/

[3] S. Sharma et al., "TransMistral at SemEval-2024 Task 10: Using Mistral 7B for Emotion Discovery and Reasoning its Flip in Conversation," in *Proc. SemEval*, 2024. [Online]. Available: https://aclanthology.org/2024.semeval-1.46.pdf
