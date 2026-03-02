# Proposed Method — Phương Pháp Đề Xuất

## Text-Only Deep Reasoning Architecture cho Phân Tích Cảm Xúc Hội Thoại

**Kiến trúc kết hợp: TransMistral (Context Engine) + RVISA (Logic Core) + MASIVE (Vocabulary Head)**

---

## 1. Tổng Quan Hệ Thống

### 1.1 Bài toán & Mục tiêu

Hệ thống giải quyết bài toán **phân tích cảm xúc đa tầng (multi-level sentiment analysis)** trên dữ liệu hội thoại văn bản (text-only), bao gồm:

- **Facebook comment threads** (thảo luận dạng cây)
- **Chat logs** (hỗ trợ khách hàng, nhắn tin)
- **Forum threads** (bài viết, trả lời, trích dẫn lồng nhau)

Ba vấn đề cốt lõi cần giải quyết:

| Vấn đề | Mô tả | Module phụ trách |
|---------|--------|------------------|
| **Context** | Xử lý lịch sử hội thoại dài, tương tác giữa các speaker, phát hiện **emotion flips** | TransMistral |
| **Logic** | Suy luận *tại sao* người dùng có cảm xúc đó (aspect → cause), hỗ trợ implicit sentiment & sarcasm | RVISA |
| **Granularity** | Sinh nhãn trạng thái cảm xúc chi tiết (1,000+ phong cách từ vựng) thay vì chỉ phân loại thô | MASIVE |

### 1.2 Cam kết sản phẩm

> Không chỉ dán nhãn **"Angry"** — mà là **"frustrated due to neglect"** (thất vọng do bị phớt lờ), kèm bằng chứng hội thoại và lập luận đã được kiểm chứng.

---

## 2. Workflow End-to-End

### 2.1 Sơ đồ luồng xử lý

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      Raw Conversation Data                         │
│              (Facebook / Chat / Forum — text-only)                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 0 — Preprocessing & Normalization                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Flatten  │→│ Speaker  │→│  Text    │→│   PII    │→│ Language │ │
│  │ Thread   │ │ Canon.   │ │ Cleaning │ │ Redact   │ │ Detect   │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
│                                                        ┌──────────┐│
│                                                        │Translate ││
│                                                        │(optional)││
│                                                        └──────────┘│
│  Output: PreprocessedConversation (Contract B)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — TransMistral: Contextual Parsing (API-based)             │
│                                                                     │
│  • Serialization hội thoại → prompt chuẩn hóa                      │
│  • Gọi Mistral API (mistral-small-latest)                          │
│  • JSON parse + validation + retry                                  │
│                                                                     │
│  Output: TransMistralOutput (Contract C)                            │
│    ├── coarse_timeline (emotion + prob per utterance)               │
│    ├── flip_events (emotion trajectory changes)                     │
│    ├── anchors[] (utterances cần deep reasoning)                    │
│    └── context_summary (tóm tắt ngữ cảnh toàn cục)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Anchor Gating  │
                    │  (Cost Control) │
                    └────────┬────────┘
                             │ chỉ anchor đã filter
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — RVISA: Reasoning & Verification (2-stage, per anchor)    │
│                                                                     │
│  ┌────────────────────┐     ┌─────────────────────────┐            │
│  │  2A — Generator    │────▶│   2B — Verifier          │            │
│  │  (Hypothesis)      │     │   (Evidence Checking)    │            │
│  │                    │     │                          │            │
│  │  aspect → cause    │     │  verdict: PASS / FAIL   │            │
│  │  → attitude        │     │  confidence score       │            │
│  │  + evidence hints  │     │  corrected reasoning    │            │
│  └────────────────────┘     └─────────────────────────┘            │
│                                                                     │
│  Output: RVISAOutput (Contract D) — only PASS results proceed      │
└────────────────────────────┬────────────────────────────────────────┘
                             │ chỉ verdict = PASS
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — MASIVE: Fine-Grained Affective Labeling                  │
│                                                                     │
│  • Input: verified rationale + aspect + cause + attitude            │
│  • Output: nhãn cảm xúc chi tiết (2-6 từ, tâm lý học)             │
│  • Normalization: exact match / embedding / clustering              │
│                                                                     │
│  Output: MASIVEOutput (Contract E)                                  │
│    ├── fine_grained_label ("frustrated due to neglect")             │
│    ├── alt_labels (["ignored", "disrespected"])                     │
│    └── confidence + normalization metadata                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FINAL ASSEMBLY — FinalEmotionReport                                │
│                                                                     │
│  • Merge coarse emotions (toàn bộ utterances)                       │
│  • Attach RVISA + MASIVE results cho các anchor                     │
│  • Tính toán dominant_states + key_anchors                          │
│  • Lưu trữ / trả về JSON report                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi Tiết Từng Bước

### 3.1 Step 0 — Ingestion + Preprocessing + Normalization

**Mục đích**: Chuyển đổi dữ liệu thô, không đồng nhất thành biểu diễn chuẩn hóa, bảo toàn tín hiệu cảm xúc.

#### Đầu vào
- `ConversationObject` (Contract A): utterances thô + metadata từ nguồn

#### Các thao tác xử lý (theo thứ tự)

| Bước | Thao tác | Chi tiết |
|------|----------|----------|
| 0.1 | **Thread Flattening** | Sắp xếp thời gian, bảo toàn `reply_to_utt_id` |
| 0.2 | **Speaker Canonicalization** | Gán `speaker_id` nhất quán (S1, S2, ...) |
| 0.3 | **Text Cleaning** | URL → `<url>`, @mention → `<user>`, normalize whitespace. **Giữ nguyên** emoji, "!!!", "???" và ký tự kéo dài |
| 0.4 | **PII Redaction** | phone → `<pii_phone>`, email → `<pii_email>` |
| 0.5 | **Language Detection** | Phát hiện `vi/en/mixed/unknown` per utterance (sử dụng `langdetect` + heuristic Unicode tiếng Việt) |
| 0.6 | **Translation** (optional) | Dịch sang pivot language (EN) nếu mixed/VI, cache theo hash `text_clean` |

#### Đầu ra
- `PreprocessedConversation` (Contract B)

#### Xử lý lỗi
- Thiếu timestamp → dùng thứ tự platform + `ordering_confidence`
- HTML/entities lỗi → decode an toàn, giữ bản raw backup
- Độ dài quá lớn → chính sách chunking cho Step 1

---

### 3.2 Step 1 — TransMistral: Contextual Parsing

**Mục đích**: Phân tích **toàn bộ ngữ cảnh hội thoại** để xác định:
- Cảm xúc thô (coarse emotion) cho mỗi utterance
- Các sự kiện **emotion flip** (thay đổi quỹ đạo cảm xúc)
- Các **anchor utterances** cần suy luận sâu
- **Context summary** toàn cục

#### Serialization hội thoại

Mỗi utterance được serialize theo format chuẩn:
```text
[utt_id=U12][t=2026-02-20T10:02:00][S=S2][reply_to=U9] raw:"..." | translated:"..."
```

#### Chiến lược xử lý hội thoại dài

| Chiến lược | Mô tả | Khi nào dùng |
|------------|--------|-------------|
| **A — Full** | Toàn bộ hội thoại | Tokens vừa đủ limit |
| **B — Sliding Tail** | Chỉ N utterances cuối | Tiết kiệm chi phí |
| **C — Summary+Tail** ★ | Tóm tắt cũ + tail window mới | **Mặc định khuyến nghị** |
| **D — Two-pass** | Pass 1: phát hiện flip zone → Pass 2: zoom refine | Chất lượng cao nhất |

#### Prompt Design

- **System prompt**: Yêu cầu trả về **ONLY valid JSON** theo schema chuẩn
- **User prompt**: 3 tasks: (1) Gán coarse emotion, (2) Phát hiện flips, (3) Chọn anchors
- **JSON repair**: Strip markdown fences, tìm `{...}` trong output, retry với prompt nghiêm ngặt hơn

#### Anchor Selection Logic

Tiêu chí chọn anchor (utterances cần suy luận sâu):
- **Flip triggers** — cảm xúc thay đổi tại đây
- **High-arousal emotions** — anger/disgust/intense sadness
- **Low-confidence** — dự đoán không chắc chắn
- **Sarcasm markers** — biểu hiện mỉa mai
- **Domain escalation** — yêu cầu hoàn tiền, đe dọa rời đi

#### Đầu ra
- `TransMistralOutput` (Contract C): `coarse_timeline`, `anchors[]`, `flip_events[]`, `context_summary`

---

### 3.3 Anchor Gating (Kiểm soát chi phí)

**Mục đích**: Chỉ chạy Step 2 + 3 trên các anchor đã được lọc, kiểm soát chi phí API.

| Tham số | Giá trị mặc định | Mô tả |
|---------|-------------------|-------|
| `anchor_threshold` | 0.65 | Ngưỡng tối thiểu anchor_score |
| `max_anchors` | 20 | Giới hạn số anchor tối đa |
| `always_include_flip_triggers` | true | Luôn giữ flip triggers dù score thấp |

---

### 3.4 Step 2 — RVISA: Reasoning & Verification

**Mục đích**: Cho mỗi anchor, suy luận *tại sao* cảm xúc tồn tại và kiểm chứng lập luận dựa trên bằng chứng văn bản.

#### Stage 2A — Generator (Đặt giả thuyết)

Đầu vào (xây dựng bởi Orchestrator):
- `anchor_utt_id` + `anchor_text`
- **Context window**: k utterances trước/sau anchor
- `context_summary` từ Step 1

Input vào prompt:
```text
Global context summary: "..."
Window:
[U10][S1] "..."
[U11][S2] "..."
Anchor:
[U12][S2] "..."
```

Output (JSON):
```json
{
  "aspect": "responsiveness",
  "cause": "ignored for 48 hours",
  "inferred_attitude": "negative",
  "rationale": "...",
  "evidence": [{"utt_id":"U10","quote":"..."}]
}
```

#### Window Building (Context Window)

| Trường hợp | `k_before` | `k_after` |
|-------------|-----------|----------|
| Mặc định | 6 | 2 |
| Gần flip zone | 12 | 4 |

#### Stage 2B — Verifier (Kiểm chứng)

Kiểm tra xem lập luận của Generator có được hỗ trợ bởi bằng chứng trong window không.

**PASS nếu tất cả đúng:**
- Aspect/cause được nêu rõ hoặc ngụ ý mạnh trong window
- Rationale không bịa ra sự kiện/thực thể mới
- Evidence spans trỏ tới nội dung text thực tế
- Quy kết speaker chính xác

**FAIL nếu bất kỳ:**
- Rationale dựa trên giả định không có căn cứ
- Mâu thuẫn với text trong window
- Sai quy kết speaker
- Sarcasm được "diễn giải" mà không có bằng chứng ngữ cảnh

#### Đầu ra
- `RVISAOutput` (Contract D): `verdict`, `confidence`, `aspect`, `cause`, `verified_rationale`, `evidence_spans[]`

---

### 3.5 Step 3 — MASIVE: Fine-Grained Labeling

**Mục đích**: Chuyển đổi lập luận đã kiểm chứng thành **nhãn trạng thái cảm xúc chi tiết** phù hợp cho:
- UX triggers (leo thang, ngăn churn, mẫu xin lỗi)
- BI dashboards (top cảm xúc theo khu vực sản phẩm)
- CRM annotations (hướng dẫn agent)

#### Ràng buộc nhãn
- 2–6 từ (English) hoặc 2–8 từ (Vietnamese)
- Tránh nhãn quá chung ("negative", "sad")
- Ưu tiên cụm từ có ý nghĩa tâm lý:
  - `"betrayed"`, `"dismissive resentment"`, `"frustrated due to neglect"`, `"anxious uncertainty"`

#### Normalization

| Phương pháp | Mô tả | Phù hợp cho |
|-------------|--------|-------------|
| **Exact match** | Khớp chính xác với vocabulary | Hệ thống có vocab sẵn |
| **Embedding match** | Nearest neighbor trong không gian embedding | Phổ quát |
| **LLM rerank** | LLM xếp hạng top N candidates | Chính xác nhất |
| **Clustering** | Nhóm nhãn thành 50–200 buckets | Khi chưa có vocab |

#### Đầu ra
- `MASIVEOutput` (Contract E): `fine_grained_label`, `alt_labels[]`, `confidence`, `normalization`

---

## 4. Data Contracts — Luồng Dữ Liệu

```text
Contract A (Raw Input)
    │
    ▼ Step 0
Contract B (PreprocessedConversation)
    │
    ▼ Step 1
Contract C (TransMistralOutput)
    │
    ├──▶ Anchor Gating
    │
    ▼ Step 2 (per anchor)
Contract D (RVISAOutput)
    │
    ▼ Step 3 (only PASS)
Contract E (MASIVEOutput)
    │
    ▼ Final Assembly
FinalEmotionReport
```

Mỗi contract giữa các bước được định nghĩa bằng **Pydantic models** trong `modules/models.py`, đảm bảo type safety và schema validation tự động.

---

## 5. Orchestrator — Bộ Điều Phối

### 5.1 Thuật toán chính

```
1. Nhận ConversationObject (raw input)
2. Step 0: Preprocess
   → flatten, clean, detect language, translate (optional), redact PII
3. Gọi Step 1: TransMistral
   → coarse_timeline + flips + anchors + context_summary
4. Anchor Gating
   → filter theo threshold + max_anchors + flip triggers
5. Với mỗi anchor:
   a. Xây dựng context window (mở rộng gần flips)
   b. Gọi RVISA Generator → RVISA Verifier
   c. Nếu PASS → Gọi MASIVE cho fine label + normalization
   d. Nếu FAIL → Fallback về coarse output của Step 1
6. Final Assembly
   → Merge Step 1 (toàn bộ) + Step 2/3 (anchors)
   → Tính dominant_states, key_anchors
7. Trả về FinalEmotionReport
```

### 5.2 LLM Backend

- **Provider**: Mistral AI (`mistral-small-latest`)
- **Giao tiếp**: REST API qua `call_llm_mistral.py`
- **Quản lý API keys**: Round-robin rotation, cooldown per key, rate-limit handling
- **Retry**: Exponential backoff + jitter, infinite retry mode

---

## 6. Reliability — Đảm Bảo Độ Tin Cậy

### 6.1 JSON Validation & Repair

Mỗi bước API output đều qua:

```text
1. Strict JSON parse
2. Schema validation (Pydantic)
3. Nếu invalid → retry với prompt nghiêm ngặt hơn
4. Nếu vẫn invalid → JSON repair (strip markdown fences, tìm {...})
5. Nếu vẫn invalid → fallback tùy step
```

### 6.2 Chính sách Fallback

| Step thất bại | Hành vi |
|---------------|---------|
| Step 1 fails | Trả về preprocessing metadata + error |
| Step 2 fails (per anchor) | Giữ coarse emotion từ Step 1, đánh dấu `unverified` |
| Step 3 fails | Giữ verified rationale, nhãn = `"unknown"` |

### 6.3 Retry & Backoff

- Retry lỗi HTTP tạm thời
- Exponential backoff + jitter
- Circuit breaker per API key để ngăn cascading failures

---

## 7. Cost & Latency Controls

| Cơ chế | Mô tả |
|--------|-------|
| **Anchor Gating** | Chỉ chạy Step 2/3 trên anchors (kiểm soát # LLM calls) |
| **Windowing** | Giới hạn context window size (tokens) |
| **Caching** | Cache theo `hash(input + model_version)` per step |
| **Observability** | Correlation ID, log latency/tokens/retries per step |

---

## 8. Kiến Trúc File Dự Án

```text
SentimentSystem/
├── call_llm_mistral.py          # LLM API interface (Mistral)
├── configs/
│   └── pipeline_config.yaml     # Cấu hình pipeline
├── data/
│   └── sample_input.json        # Dữ liệu mẫu (Contract A)
├── modules/
│   ├── models.py                # Pydantic data contracts (A–E + Final)
│   ├── preprocessing/
│   │   └── engine.py            # Step 0: Preprocessing
│   ├── transmistral/
│   │   └── engine.py            # Step 1: TransMistral
│   ├── rvisa/
│   │   └── engine.py            # Step 2: RVISA (Generator + Verifier)
│   └── masive/
│       └── engine.py            # Step 3: MASIVE
└── scripts/
    └── run_pipeline.py          # Orchestrator (CLI entry point)
```

---

## 9. Ví Dụ Walkthrough End-to-End

**Kịch bản**: Chat hỗ trợ khách hàng 7 lượt, khách hàng chuyển từ bình tĩnh → bực bội → thất vọng.

### Step 0 — Preprocessing
- Ingest chat export → chuẩn hóa utterances (U1–U7)
- Phát hiện ngôn ngữ: Vietnamese (`vi`)
- Redact PII: đơn hàng `#12345` → `<pii_phone>` (nếu match pattern)
- Text clean: giữ nguyên `"???"` và `"..."` (tín hiệu cảm xúc)

### Step 1 — TransMistral
- **Coarse timeline**: U1(neutral) → U2(neutral) → U3(**anger**, 0.87, flip) → U4(neutral) → U5(**anger**, 0.91) → U6(neutral) → U7(**sadness**, 0.72)
- **Flip event**: U2→U3 (`calm→anger`), trigger: U3
- **Anchors**: U3 (flip-trigger, score=0.85), U5 (high-arousal, score=0.81), U7 (mixed-signal, score=0.70)

### Anchor Gating
- Threshold 0.65 → giữ cả 3 anchors
- U3 là flip trigger → luôn giữ

### Step 2 — RVISA (cho U5)
- **Window**: [U2, U3, U4, U5, U6, U7] (mở rộng vì gần flip)
- **Generator**: aspect="service reliability", cause="repeated broken promises about checking", attitude="negative"
- **Verifier**: evidence = U3("đã chờ 2 tuần"), U5("Lần nào cũng nói kiểm tra") → **PASS**, confidence=0.82

### Step 3 — MASIVE (cho U5)
- Input: verified rationale từ Step 2
- **Output**: `"frustrated due to broken promises"` (alt: `"exasperated disappointment"`, `"exhausted patience"`)
- Confidence: 0.78

### Final Report
- Timeline đầy đủ 7 utterances với coarse emotions
- 3 anchors có verified reasoning + fine labels + evidence
- **Dominant state**: `"frustrated due to broken promises"`

---

## 10. Tài Liệu Tham Khảo

| Nguồn | Link |
|-------|------|
| **RVISA Paper** | [PolyU Repository](https://ira.lib.polyu.edu.hk/bitstream/10397/112805/1/Lai_RVISA_Reasoning_Verification.pdf) |
| **RVISA Landing** | [EdUHK Publications](https://repository.eduhk.hk/en/publications/rvisa-reasoning-and-verification-for-implicit-sentiment-analysis/) |
| **MASIVE (ACL)** | [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1139/) |
| **MASIVE Paper** | [PDF](https://aclanthology.org/2024.emnlp-main.1139.pdf) |
| **MASIVE arXiv** | [arXiv 2407.12196](https://arxiv.org/abs/2407.12196) |
| **MASIVE Code** | [GitHub](https://github.com/NickDeas/MASIVE) |
| **TransMistral** | [SemEval Paper](https://aclanthology.org/2024.semeval-1.46.pdf) |
| **TransMistral** | [ResearchGate](https://www.researchgate.net/publication/382633282) |
