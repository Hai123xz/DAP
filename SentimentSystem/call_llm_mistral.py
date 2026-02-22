
"""
LLM API Interface for EMERGE - Mistral Integration

This module provides text and vision (multimodal) helpers for Mistral API:
- Named Entity Recognition (NER) from clinical notes
- Summary generation from EHR + notes + KG context
- ECG image analysis via multimodal vision API

Merged from call_llm_mistral_update.py:
- Smart API key cooldown + rate-limit tracking
- Capacity tier monitoring
- Per-key success tracking
- Truly infinite retry (no cap)
"""

import base64
import json as _json
import logging
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Optional, Union

import requests
from dotenv import load_dotenv

# ========================= CONFIG =========================
load_dotenv()

logger = logging.getLogger(__name__)

MISTRAL_API_KEYS = [
    os.getenv("MISTRAL_API_KEY"),
    os.getenv("MISTRAL_API_KEY_0"),
    os.getenv("MISTRAL_API_KEY_1"),
    os.getenv("MISTRAL_API_KEY_2"),
    os.getenv("MISTRAL_API_KEY_3"),
    os.getenv("MISTRAL_API_KEY_4"),
    os.getenv("MISTRAL_API_KEY_5"),
    os.getenv("MISTRAL_API_KEY_6"),
    os.getenv("MISTRAL_API_KEY_7"),
    os.getenv("MISTRAL_API_KEY_8"),
    os.getenv("MISTRAL_API_KEY_9"),
    os.getenv("MISTRAL_API_KEY_10"),
    os.getenv("MISTRAL_API_KEY_11"),
    os.getenv("MISTRAL_API_KEY_12"),
    os.getenv("MISTRAL_API_KEY_13"),
    os.getenv("MISTRAL_API_KEY_14"),
    os.getenv("MISTRAL_API_KEY_15"),
    os.getenv("MISTRAL_API_KEY_16"),
    os.getenv("MISTRAL_API_KEY_17"),
    os.getenv("MISTRAL_API_KEY_18"),
    os.getenv("MISTRAL_API_KEY_19"),
    os.getenv("MISTRAL_API_KEY_20"),
    os.getenv("MISTRAL_API_KEY_21"),
    os.getenv("MISTRAL_API_KEY_22"),
    os.getenv("MISTRAL_API_KEY_23"),
    os.getenv("MISTRAL_API_KEY_24"),
    os.getenv("MISTRAL_API_KEY_25"),
    os.getenv("MISTRAL_API_KEY_26"),
]
MISTRAL_API_KEYS = [k for k in MISTRAL_API_KEYS if k]  # Remove None values

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL_NAME = "mistral-small-latest"   # ← kept as-is
MAGISTRAL_MODEL_NAME = "mistral-small-latest"  # ← kept as-is

# Request timeout in seconds (prevents hanging)
REQUEST_TIMEOUT = 180

# ========================= KEY MANAGEMENT =========================
# API key timeout tracking (unified for both cooldown and rate limits)
RATE_LIMIT_TIMEOUT = 180   # 30 min for rate-limit errors
COOLDOWN_TIMEOUT = 30                 # 30 s cooldown after each use
_api_key_timeouts = {}                # {api_key_index: timeout_until_timestamp}
_api_key_rate_limited = set()         # Track which keys are rate-limited
_timeout_lock = Lock()

# Round-robin API key selector
_api_rr_counter = [0]
_api_rr_lock = Lock()

# Per-key success tracking
_api_key_success_total = {}   # {key_index: count}
_api_key_success_24h = {}     # {key_index: deque of timestamps}
_success_lock = Lock()

# Capacity tier tracking (thresholds: 100, 75, 50, 25, 0)
_current_capacity_tier = 100

# Status log path
STATUS_LOG = "api_keys/status.jsonl"

# ========================= HELPERS – LOG =========================

def _log_to_file(path: str, line: str):
    """Append a single line to a log file (create dir if needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(line + "\n")


def _record_success(api_key_index: int):
    """Record a successful API call for a key."""
    now = time.time()
    with _success_lock:
        _api_key_success_total[api_key_index] = (
            _api_key_success_total.get(api_key_index, 0) + 1
        )
        if api_key_index not in _api_key_success_24h:
            _api_key_success_24h[api_key_index] = deque()
        _api_key_success_24h[api_key_index].append(now)
        cutoff = now - 86400
        while (
            _api_key_success_24h[api_key_index]
            and _api_key_success_24h[api_key_index][0] < cutoff
        ):
            _api_key_success_24h[api_key_index].popleft()


def _get_key_stats() -> dict:
    """Get per-key success stats: {key_str: [total, last_24h]}."""
    now = time.time()
    cutoff = now - 86400
    stats = {}
    with _success_lock:
        for idx in range(len(MISTRAL_API_KEYS)):
            total = _api_key_success_total.get(idx, 0)
            recent = sum(
                1 for t in _api_key_success_24h.get(idx, deque()) if t > cutoff
            )
            stats[str(idx)] = [total, recent]
    return stats


# ========================= CAPACITY TIER =========================

def _compute_capacity_tier() -> int:
    """Compute capacity tier. Must hold _timeout_lock."""
    total = len(MISTRAL_API_KEYS)
    if total == 0:
        return 0
    now = time.time()
    rate_limited = sum(
        1
        for idx in _api_key_rate_limited
        if idx in _api_key_timeouts and _api_key_timeouts[idx] > now
    )
    pct = (total - rate_limited) / total * 100
    if pct > 75:
        return 100
    elif pct > 50:
        return 75
    elif pct > 25:
        return 50
    elif pct > 0:
        return 25
    else:
        return 0


def _check_and_log_capacity_change():
    """Check if capacity tier changed and log if so."""
    global _current_capacity_tier
    event = None
    with _timeout_lock:
        new_tier = _compute_capacity_tier()
        old_tier = _current_capacity_tier
        if new_tier != old_tier:
            _current_capacity_tier = new_tier
            event = "capacity_drop" if new_tier < old_tier else "capacity_rise"
    if event:
        write_status_log(event)


def _get_key_status():
    """Get current active/inactive key status. Must hold _timeout_lock."""
    active = []
    inactive = []
    now = time.time()
    min_recovery_secs = 0
    for idx in range(len(MISTRAL_API_KEYS)):
        if idx in _api_key_rate_limited and idx in _api_key_timeouts:
            remaining = _api_key_timeouts[idx] - now
            if remaining > 0:
                inactive.append({
                    "key": idx,
                    "recovery": datetime.fromtimestamp(
                        _api_key_timeouts[idx]
                    ).strftime("%H:%M:%S"),
                    "secs_left": round(remaining),
                })
                if min_recovery_secs == 0 or remaining < min_recovery_secs:
                    min_recovery_secs = remaining
                continue
        active.append(idx)
    return active, inactive, round(min_recovery_secs)


def write_status_log(event: str):
    """Write a one-line JSON status to the log file."""
    with _timeout_lock:
        active, inactive, next_recovery_secs = _get_key_status()
        capacity_tier = _current_capacity_tier
    key_stats = _get_key_stats()
    status = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "capacity_pct": capacity_tier,
        "active_keys": active,
        "inactive_keys": inactive,
        "next_recovery_secs": next_recovery_secs,
        "key_stats": key_stats,
    }
    _log_to_file(STATUS_LOG, _json.dumps(status, separators=(",", ":")))


# ========================= KEY AVAILABILITY =========================

def mark_api_key_timeout(api_key_index: int, duration: int = RATE_LIMIT_TIMEOUT):
    """Mark an API key as timed out for a specified duration."""
    with _timeout_lock:
        _api_key_timeouts[api_key_index] = time.time() + duration
        if duration >= RATE_LIMIT_TIMEOUT:
            _api_key_rate_limited.add(api_key_index)
    _check_and_log_capacity_change()


def mark_api_key_used(api_key_index: int):
    """Mark an API key as just used (cooldown to reserve from other threads)."""
    with _timeout_lock:
        # Only set cooldown if key isn't already rate-limited
        if api_key_index not in _api_key_rate_limited:
            _api_key_timeouts[api_key_index] = time.time() + COOLDOWN_TIMEOUT


def is_api_key_available(api_key_index: int) -> tuple:
    """Check if an API key is available. Returns (available, recovered)."""
    with _timeout_lock:
        if api_key_index not in _api_key_timeouts:
            return True, False
        if time.time() >= _api_key_timeouts[api_key_index]:
            del _api_key_timeouts[api_key_index]
            recovered = api_key_index in _api_key_rate_limited
            _api_key_rate_limited.discard(api_key_index)
            return True, recovered
        return False, False


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if an exception is a rate limit or too many requests error."""
    if isinstance(exception, requests.exceptions.HTTPError):
        if exception.response is not None and exception.response.status_code == 429:
            return True
    error_str = str(exception).lower()
    return "rate limit" in error_str or "too many requests" in error_str


# ========================= ROUND-ROBIN KEY SELECTOR =========================

def get_next_api_key() -> int:
    """Get next available API key index, skipping timed-out keys."""
    while True:
        found_idx = None
        recovered_keys = []
        min_wait_time = float("inf")

        with _api_rr_lock:
            attempts = 0
            while attempts < len(MISTRAL_API_KEYS):
                idx = _api_rr_counter[0] % len(MISTRAL_API_KEYS)
                _api_rr_counter[0] += 1
                available, recovered = is_api_key_available(idx)
                if recovered:
                    recovered_keys.append(idx)
                if available:
                    found_idx = idx
                    break
                attempts += 1

            if found_idx is None:
                with _timeout_lock:
                    now = time.time()
                    for i in range(len(MISTRAL_API_KEYS)):
                        if i in _api_key_timeouts:
                            wait = _api_key_timeouts[i] - now
                            if 0 < wait < min_wait_time:
                                min_wait_time = wait

        # Check capacity tier on recovery
        if recovered_keys:
            _check_and_log_capacity_change()

        if found_idx is not None:
            return found_idx

        # Sleep outside the lock to avoid deadlock
        if min_wait_time < float("inf") and min_wait_time > 0:
            logger.info("All keys busy, waiting %.1fs for next key…", min_wait_time)
            time.sleep(min_wait_time)
        else:
            time.sleep(10)


# ========================= IMAGE HELPERS =========================

_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    Read an image file and return its Base64-encoded data URI.

    Args:
        image_path: Path to the image file (png, jpg, webp, etc.)

    Returns:
        Base64 data URI string, e.g. "data:image/png;base64,iVBOR..."

    Raises:
        FileNotFoundError: if the path does not exist
        ValueError: if the file extension is not a supported image type
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    ext = p.suffix.lower()
    mime = _MIME_MAP.get(ext)
    if mime is None:
        raise ValueError(
            f"Unsupported image extension '{ext}'. "
            f"Supported: {list(_MIME_MAP.keys())}"
        )

    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def remove_reasoning(response_content: str) -> str:
    """Remove <think>...</think> reasoning block if present."""
    match = re.search(r"</think>\s*(.*)", response_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_content.strip()


# ========================= CORE API CALL =========================

def _make_api_call(
    payload: dict,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """
    Send payload to Mistral API using a single round-robin key.
    Marks key with cooldown before call and rate-limit on 429.

    Returns:
        Raw content string from the response.
    """
    api_idx = get_next_api_key()
    api_key = MISTRAL_API_KEYS[api_idx]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Reserve key BEFORE the call so other threads skip it
    mark_api_key_used(api_idx)

    try:
        response = requests.post(
            MISTRAL_API_URL, headers=headers, json=payload, timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        # Re-apply cooldown AFTER success
        mark_api_key_used(api_idx)
        _record_success(api_idx)
        return data["choices"][0]["message"]["content"] or ""
    except requests.exceptions.Timeout:
        logger.warning("Timeout with API key #%d after %ds", api_idx, timeout)
        raise
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.warning("HTTP %s with API key #%d: %s", status, api_idx, e)
        if is_rate_limit_error(e):
            mark_api_key_timeout(api_idx)
        raise
    except Exception as e:
        logger.warning("Error with API key #%d: %s", api_idx, e)
        raise


def _retry_wrapper(call_fn, infinite_retry: bool) -> str:
    """Wrap a callable with optional retry logic.

    When infinite_retry is True, retries indefinitely (no cap)
    with exponential backoff + jitter to avoid thundering herd.
    """
    import random

    if not infinite_retry:
        return remove_reasoning(call_fn())

    base_delay = 2        # initial backoff in seconds
    max_delay = 120       # cap at 2 minutes
    attempt = 0

    while True:
        try:
            result = remove_reasoning(call_fn())
            return result
        except Exception as e:
            attempt += 1
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            sleep_time = delay + jitter
            logger.info(
                "Retry #%d after error (sleeping %.1fs): %s",
                attempt, sleep_time, e,
            )
            time.sleep(sleep_time)


# ========================= TEXT API =========================

def ask(
    user_prompt: str,
    sys_prompt: str = "",
    model_name: str = MISTRAL_MODEL_NAME,
    max_tokens: int = 32768,
    temperature: float = 0.1,
    infinite_retry: bool = False,
) -> str:
    """
    Send a **text-only** prompt to the Mistral chat API.

    Args:
        user_prompt: The user message.
        sys_prompt:  Optional system message.
        model_name:  Mistral model identifier.
        max_tokens:  Maximum tokens in the response.
        temperature: Sampling temperature.
        infinite_retry: If True, retry indefinitely on failure.

    Returns:
        Generated text (reasoning blocks stripped).
    """

    def _call_once() -> str:
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return _make_api_call(payload)

    return _retry_wrapper(_call_once, infinite_retry)


def ask_with_messages(
    messages: list,
    model_name: str = MISTRAL_MODEL_NAME,
    max_tokens: int = 32768,
    temperature: float = 0.1,
    infinite_retry: bool = False,
) -> str:
    """
    Send a structured message array to the Mistral chat API.

    Args:
        messages:  List of message dicts with "role" and "content" keys.
        model_name:    Mistral model identifier.
        max_tokens:    Maximum tokens in the response.
        temperature:   Sampling temperature.
        infinite_retry: If True, retry indefinitely on failure.

    Returns:
        Generated text (reasoning blocks stripped).
    """

    def _call_once() -> str:
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return _make_api_call(payload)

    return _retry_wrapper(_call_once, infinite_retry)


# ========================= VISION (MULTIMODAL) API =========================

def ask_vision(
    user_prompt: str,
    image_sources: Union[str, Path, List[Union[str, Path]]],
    sys_prompt: str = "",
    model_name: str = MISTRAL_MODEL_NAME,
    max_tokens: int = 32768,
    temperature: float = 0.1,
    infinite_retry: bool = False,
) -> str:
    """
    Send a **multimodal** prompt (text + images) to the Mistral chat API.

    Each image can be specified as:
      - A local file path  → auto-encoded to base64 data URI
      - A URL string starting with "http"
      - A base64 data URI string ("data:image/...")

    Args:
        user_prompt:   Text portion of the prompt.
        image_sources: Single image or list of images (path / URL / data URI).
        sys_prompt:    Optional system message.
        model_name:    Mistral model identifier.
        max_tokens:    Maximum tokens in the response.
        temperature:   Sampling temperature.
        infinite_retry: If True, retry indefinitely on failure.

    Returns:
        Generated text (reasoning blocks stripped).

    Example::

        response = ask_vision(
            user_prompt="Phân tích ảnh ECG này",
            image_sources="gradcam_out/overlay_idx_42.png",
        )
    """
    # Normalise to list
    if isinstance(image_sources, (str, Path)):
        image_sources = [image_sources]

    def _resolve_image(src: Union[str, Path]) -> dict:
        """Convert a single image source to the Mistral content block."""
        src_str = str(src)

        # Already a data URI
        if src_str.startswith("data:image/"):
            return {
                "type": "image_url",
                "image_url": {"url": src_str},
            }

        # Remote URL
        if src_str.startswith("http://") or src_str.startswith("https://"):
            return {
                "type": "image_url",
                "image_url": {"url": src_str},
            }

        # Local file path → encode to base64
        data_uri = encode_image_to_base64(src_str)
        return {
            "type": "image_url",
            "image_url": {"url": data_uri},
        }

    def _call_once() -> str:
        # Build multimodal content array
        content: list = [{"type": "text", "text": user_prompt}]
        for img in image_sources:
            content.append(_resolve_image(img))

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return _make_api_call(payload)

    return _retry_wrapper(_call_once, infinite_retry)


# ========================= COMPATIBILITY CLASS =========================

class KGSumLLM_mistral:
    def __init__(self, model_name: str = MISTRAL_MODEL_NAME):
        self.model_name = model_name

    def chat(
        self,
        messages,
        model_name: str = MISTRAL_MODEL_NAME,
        max_tokens: int = 32768,
        infinite_retry: bool = True,
    ):
        """Compatibility shim for callers expecting llm.chat(messages)."""

        def _extract_content(msg) -> str:
            # Prefer direct content if available and non-empty
            if hasattr(msg, "content") and msg.content:
                return str(msg.content)

            # llama_index ChatMessage may carry text blocks
            if hasattr(msg, "blocks") and msg.blocks:
                texts = []
                for block in msg.blocks:
                    if hasattr(block, "text") and block.text:
                        texts.append(str(block.text))
                    elif isinstance(block, dict) and block.get("text"):
                        texts.append(str(block.get("text")))
                if texts:
                    return "\n".join(texts)

            # Dict-based messages
            if isinstance(msg, dict):
                content_val = msg.get("content")
                if content_val:
                    return str(content_val)
                blocks_val = msg.get("blocks")
                if blocks_val:
                    texts = []
                    for block in blocks_val:
                        if isinstance(block, dict) and block.get("text"):
                            texts.append(str(block.get("text")))
                    if texts:
                        return "\n".join(texts)

            return ""

        def _normalize_message(message) -> dict:
            def _role_to_str(role_obj) -> str:
                if hasattr(role_obj, "value"):
                    return str(role_obj.value)
                return str(role_obj)

            if hasattr(message, "role"):
                role_val = _role_to_str(message.role)
            elif isinstance(message, dict):
                role_val = _role_to_str(message.get("role", "user"))
            else:
                role_val = "user"

            content_val = _extract_content(message)
            if not content_val:
                content_val = str(message)

            return {"role": role_val, "content": content_val}

        normalized_messages = [_normalize_message(msg) for msg in (messages or [])]

        def _call_once() -> str:
            payload = {
                "model": model_name or self.model_name,
                "messages": normalized_messages,
                "max_tokens": max_tokens,
            }
            return _make_api_call(payload)

        return _retry_wrapper(_call_once, infinite_retry)


# ========================= TEST =========================

if __name__ == "__main__":
    """Test LLM API functionality"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=" * 60)
    print("Testing Mistral Large Integration")
    print(f"  Model : {MISTRAL_MODEL_NAME}")
    print(f"  Keys  : {len(MISTRAL_API_KEYS)} loaded")
    print("=" * 60)

    # --- Test 1: text-only ---
    print("\n[Test 1] Text-only ask()...")
    try:
        start = time.time()
        response = ask(
            user_prompt="Why is the sky blue? Answer in one sentence.",
            max_tokens=256,
        )
        elapsed = time.time() - start
        print(f"  ✔ Response ({elapsed:.2f}s): {response[:200]}")
    except Exception as e:
        print(f"  ✖ Failed: {e}")

    # --- Test 2: vision (only if a sample image exists) ---
    sample_img = Path("gradcam_out")
    if sample_img.exists():
        imgs = list(sample_img.glob("*.png"))[:1] or list(sample_img.glob("*.jpg"))[:1]
        if imgs:
            print(f"\n[Test 2] Vision ask_vision() with {imgs[0].name}...")
            try:
                start = time.time()
                response = ask_vision(
                    user_prompt="Describe what you see in this image in one sentence.",
                    image_sources=imgs[0],
                    max_tokens=256,
                )
                elapsed = time.time() - start
                print(f"  ✔ Response ({elapsed:.2f}s): {response[:200]}")
            except Exception as e:
                print(f"  ✖ Failed: {e}")
        else:
            print("\n[Test 2] Skipped — no images found in gradcam_out/")
    else:
        print("\n[Test 2] Skipped — gradcam_out/ directory not found")

    print("\nDone.")
