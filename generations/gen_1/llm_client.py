"""
llm_client.py — Multi-provider LLM client (Ollama + Groq + Google Gemini).

Provider priority:
  0. Ollama (OLLAMA_BASE_URL set)  — local model, no rate limits, highest priority
  1. Groq (GROQ_API_KEY)           — llama-4-scout → llama-3.1-8b → llama-3.3-70b
  2. Gemini (GEMINI_API_KEY)       — gemini-2.0-flash → gemini-2.0-flash-lite
     Auto-activated when all Groq models are exhausted.

Rate-limit handling:
  TPM limit (short wait ≤15 min) → sleep, retry same model.
  TPD limit (long wait  >15 min) → switch to next model in chain.
  All Groq exhausted + Gemini key present → switch to Gemini provider.
  All providers exhausted → wait 30 min, reset, retry.
  Ollama: no rate limits — retried on connection error only.

Token tracking written to data/token_usage.json after every call.
Model status (active, exhausted, unavailable) written to data/model_status.json.

Ollama setup (Google Colab / local):
  export OLLAMA_BASE_URL=http://localhost:11434
  export OLLAMA_MODEL=qwen2.5-coder:14b   # optional, defaults below
"""
import concurrent.futures
import datetime
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ── Ollama (local, no rate limits) ───────────────────────────────────────────
# Set OLLAMA_BASE_URL to enable. Falls back to Groq/Gemini if not reachable.
_OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "")   # e.g. http://localhost:11434
_OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "qwen3:14b")
_OLLAMA_CALL_TIMEOUT = 180  # Qwen3 thinking mode can take longer; 3 min cap

# ── Groq models (confirmed working on free tier) ──────────────────────────────
_GROQ_LIMITS: Dict[str, Dict[str, int]] = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {"tpd": 500_000, "tpm": 30_000},
    "llama-3.1-8b-instant":                       {"tpd": 500_000, "tpm": 20_000},
    "llama-3.3-70b-versatile":                    {"tpd": 100_000, "tpm": 12_000},
}
_GROQ_FALLBACK_ORDER = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # groq fallback1 (500k TPD / 30k TPM)
    "llama-3.1-8b-instant",                        # groq fallback2 (500k TPD / 20k TPM)
    "llama-3.3-70b-versatile",                     # groq fallback3 (100k TPD / 12k TPM)
]

# ── Gemini models (Google AI free tier) ───────────────────────────────────────
# Free limits: 1,500 RPD / 15 RPM per model (NOT a rolling window — calendar day)
# gemini-1.5-flash and gemini-1.5-flash-8b are deprecated as of 2026 — removed.
_GEMINI_LIMITS: Dict[str, Dict[str, int]] = {
    "gemini-2.0-flash":      {"tpd": 1_500,  "tpm": 15},  # tpd=RPD, tpm=RPM
    "gemini-2.0-flash-lite": {"tpd": 1_500,  "tpm": 30},  # higher RPM
}
_GEMINI_FALLBACK_ORDER = [
    "gemini-2.0-flash",       # primary — best quality
    "gemini-2.0-flash-lite",  # fallback — faster/cheaper
]

# Gemini RPM sleep: when rate-limited and can't parse retry time, sleep one RPM window
_GEMINI_RPM_SLEEP_SEC = 65   # 60s window + 5s buffer

_ALL_MODEL_LIMITS = {**_GROQ_LIMITS, **_GEMINI_LIMITS}
_DEFAULT_LIMITS   = {"tpd": 500_000, "tpm": 30_000}
_ALL_MODELS       = _GROQ_FALLBACK_ORDER + _GEMINI_FALLBACK_ORDER

# Rate-limit thresholds
_MAX_TPM_SLEEP_SEC  = 900   # sleeps longer than this → treat as TPD, switch model
_RECOVERY_WAIT_SEC  = 1800  # wait when ALL providers exhausted (30 min)
_MAX_RECOVERY_ATTEMPTS = 4  # give up after 4 × 30 min = 2 hours


def _is_gemini_model(model_name: str) -> bool:
    return model_name.startswith("gemini")


def _is_ollama_model(model_name: str) -> bool:
    """Model names that route to Ollama — anything not groq/gemini when Ollama is enabled."""
    return bool(_OLLAMA_BASE_URL) and not _is_gemini_model(model_name) and model_name not in _GROQ_FALLBACK_ORDER


def _check_ollama_reachable() -> bool:
    """Quick TCP probe to see if Ollama server is up."""
    if not _OLLAMA_BASE_URL:
        return False
    try:
        import urllib.request
        urllib.request.urlopen(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


class LLMClient:
    """
    Unified LLM client supporting Groq and Google Gemini.

    Same .call() / .chat_completion() interface — no agent changes needed.
    Falls back automatically: Groq primary → Groq fallbacks → Gemini fallbacks.
    """

    _usage_lock = threading.Lock()

    def __init__(self, config):
        self.config = config
        self._groq_key   = os.getenv("GROQ_API_KEY", "")
        self._gemini_key = (
            os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
            or os.getenv("GoogleAPIKey", "")
        )

        # ── Ollama check (highest priority — overrides all API providers) ──────
        self._ollama_url   = _OLLAMA_BASE_URL
        self._ollama_model = _OLLAMA_MODEL
        self._use_ollama   = bool(self._ollama_url) and _check_ollama_reachable()

        if not self._use_ollama and not self._groq_key and not self._gemini_key:
            raise EnvironmentError(
                "No LLM provider available. Set one of: "
                "OLLAMA_BASE_URL, GROQ_API_KEY, GEMINI_API_KEY"
            )

        # Hard cap per model attempt (Ollama gets longer since local GPU is slower)
        self._call_timeout = _OLLAMA_CALL_TIMEOUT if self._use_ollama else 30

        # Track exhausted/unavailable models across both providers
        self._exhausted_models: set = set()
        self._tpd_exhausted:    set = set()
        self._unavailable:      set = set()
        self._retry_after: Dict[str, str] = {}

        # Token usage tracking
        self._usage_date:   str  = datetime.date.today().isoformat()
        self._per_agent:    Dict = {}
        self._total_tokens: int  = 0
        self._limits = _ALL_MODEL_LIMITS.get(config.model_name, _DEFAULT_LIMITS)

        if self._use_ollama:
            self._provider = "ollama"
        elif _is_gemini_model(config.model_name):
            self._provider = "gemini"
        else:
            self._provider = "groq"

        # Resolve project root
        gen_dir = Path(__file__).resolve().parent
        project_root = (
            gen_dir.parent.parent
            if gen_dir.parent.name == "generations"
            else gen_dir.parent
        )
        self._token_file  = project_root / "data" / "token_usage.json"
        self._status_file = project_root / "data" / "model_status.json"

        # Initialise the LLM backend
        active_model = self._ollama_model if self._use_ollama else config.model_name
        self._llm = self._build_llm(active_model)

        if self._use_ollama:
            logger.info(
                f"[llm_client] Initialised — provider=ollama "
                f"model={self._ollama_model} url={self._ollama_url} "
                f"| No rate limits"
            )
        else:
            logger.info(
                f"[llm_client] Initialised — provider={self._provider} "
                f"model={config.model_name} "
                f"budget={self._limits['tpd']:,} TPD"
                + (f" / {self._limits['tpm']:,} TPM" if self._provider == "groq" else " / 15 RPM")
                + (f" | Gemini fallback: {'enabled' if self._gemini_key else 'no key'}")
            )
        self._flush_model_status()

    # ── LLM factory ──────────────────────────────────────────────────────────

    def _build_llm(self, model_name: str):
        """Build the right LangChain LLM object for the given model name."""
        # ── Ollama (local, OpenAI-compatible API) ─────────────────────────────
        if self._use_ollama:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai required for Ollama support: "
                    "pip install langchain-openai"
                )
            return ChatOpenAI(
                model=self._ollama_model,
                base_url=f"{self._ollama_url}/v1",
                api_key="ollama",          # Ollama ignores the key but field is required
                temperature=self.config.temperature,
                max_tokens=8192,
                timeout=_OLLAMA_CALL_TIMEOUT,
                max_retries=1,
            )
        # ── Gemini ────────────────────────────────────────────────────────────
        elif _is_gemini_model(model_name):
            if not self._gemini_key:
                raise EnvironmentError(f"GEMINI_API_KEY not set — cannot use {model_name}")
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self._gemini_key,
                temperature=self.config.temperature,
                max_output_tokens=8192,
                timeout=90,        # 90s hard cap — never hang longer than this
                max_retries=1,
                transport="rest",  # Force HTTP/REST — gRPC can hang on HF Spaces
            )
        # ── Groq ──────────────────────────────────────────────────────────────
        else:
            if not self._groq_key:
                raise EnvironmentError(f"GROQ_API_KEY not set — cannot use {model_name}")
            return ChatGroq(
                model=model_name,
                temperature=self.config.temperature,
                api_key=self._groq_key,
                max_tokens=8192,
                timeout=90,        # 90s hard cap — never hang longer than this
                max_retries=1,
            )

    # ── Public interface ──────────────────────────────────────────────────────

    def call(self, prompt: str, agent_name: str = "") -> str:
        """Send a single user prompt and return the response text."""
        logger.debug(f"[llm_client] call() agent={agent_name or '?'} len={len(prompt)}")
        return self._invoke_with_retry([HumanMessage(content=prompt)], agent_name or "unknown")

    def chat_completion(
        self,
        messages: List[Dict],
        response_format: Optional[Dict] = None,
        agent_name: str = "",
    ) -> str:
        """Multi-turn chat interface."""
        lc_messages = []
        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        return self._invoke_with_retry(lc_messages, agent_name or "unknown")

    # ── Retry / fallback logic ────────────────────────────────────────────────

    def _qwen3_inject_mode(self, lc_messages, agent_name: str):
        """
        Qwen3 has a built-in thinking mode controlled by /think or /no_think prefix.
        - Analyst, Architect, MetaArchitect → /think  (deep reasoning needed)
        - Coder, Critic, Reviser, Debugger  → /no_think (fast code output)
        - All others                         → /no_think (safe default)
        Strips the tag from the response before returning.
        """
        _THINK_AGENTS    = {"analyst", "architect", "meta_architect", "evolution_engine"}
        _NO_THINK_AGENTS = {"coder", "critic", "reviser", "debugger", "test_writer", "executor"}
        use_think = agent_name.lower() in _THINK_AGENTS

        prefix = "/think\n" if use_think else "/no_think\n"
        patched = []
        for msg in lc_messages:
            if isinstance(msg, HumanMessage) and not msg.content.startswith("/think"):
                patched.append(HumanMessage(content=prefix + msg.content))
            else:
                patched.append(msg)
        return patched

    def _invoke_with_retry(self, lc_messages, agent_name: str) -> str:
        # ── Ollama fast-path: no rate limits, no fallback chain needed ─────────
        if self._use_ollama:
            # Inject /think or /no_think for Qwen3 models
            if self._ollama_model.startswith("qwen3"):
                lc_messages = self._qwen3_inject_mode(lc_messages, agent_name)
            for attempt in range(3):
                try:
                    def _llm_call(llm, msgs):
                        resp = llm.invoke(msgs)
                        # Strip Qwen3 <think>...</think> block from output
                        content = resp.content
                        import re as _re
                        content = _re.sub(r'<think>.*?</think>', '', content,
                                          flags=_re.DOTALL).strip()
                        return content, resp
                    _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    _fut = _ex.submit(_llm_call, self._llm, lc_messages)
                    try:
                        content, response = _fut.result(timeout=self._call_timeout)
                        _ex.shutdown(wait=False)
                    except concurrent.futures.TimeoutError:
                        _ex.shutdown(wait=False)
                        raise TimeoutError(
                            f"Ollama call timed out after {self._call_timeout}s "
                            f"(model={self._ollama_model})"
                        )
                    # Record usage (best effort — Ollama may not return token counts)
                    try:
                        self._record_usage(agent_name, response)
                    except Exception:
                        pass
                    return content
                except Exception as exc:
                    err_str = str(exc)
                    logger.warning(
                        f"[llm_client] Ollama attempt {attempt+1}/3 failed: {err_str[:120]}"
                    )
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        raise RuntimeError(
                            f"Ollama unreachable after 3 attempts: {err_str}"
                        ) from exc
            # Should never reach here
            raise RuntimeError("Ollama invocation failed unexpectedly")

        all_models = self._build_fallback_chain()

        for recovery_attempt in range(_MAX_RECOVERY_ATTEMPTS):
            for attempt in range(len(all_models) + 4):
                try:
                    # Wrap invoke() AND response.content in the thread.
                    # response.content can block on streaming responses if accessed
                    # in the main thread — must be inside the timed thread too.
                    def _llm_call(llm, msgs):
                        resp = llm.invoke(msgs)
                        return resp.content, resp  # return both text and obj for usage

                    _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    _fut = _ex.submit(_llm_call, self._llm, lc_messages)
                    try:
                        content, response = _fut.result(timeout=self._call_timeout)
                        _ex.shutdown(wait=False)
                    except concurrent.futures.TimeoutError:
                        _ex.shutdown(wait=False)  # release main thread immediately
                        raise TimeoutError(
                            f"LLM call timed out after {self._call_timeout}s "
                            f"(model={self.config.model_name})"
                        )
                    self._record_usage(agent_name, response)
                    return content

                except Exception as exc:
                    err_str = str(exc)
                    is_rate_limit = (
                        "429" in err_str
                        or "rate_limit_exceeded" in err_str
                        or "RateLimitError" in type(exc).__name__
                        or "quota" in err_str.lower()
                        or "resource_exhausted" in err_str.lower()
                    )
                    is_timeout = (
                        "timeout" in err_str.lower()
                        or "timed out" in err_str.lower()
                        or type(exc).__name__ in ("TimeoutError", "ReadTimeout",
                                                   "ConnectTimeout", "HTTPStatusError")
                    )
                    is_unavailable = (
                        not is_timeout and (
                        "404" in err_str
                        or "model_not_found" in err_str
                        or "does not exist" in err_str
                        or "model_decommissioned" in err_str
                        or "decommissioned" in err_str.lower()
                        or "no longer supported" in err_str.lower()
                        or "not found" in err_str.lower()
                        )
                    )

                    current_model = self.config.model_name

                    if is_timeout:
                        logger.warning(
                            f"[llm_client] Timeout on {current_model} after 90s — "
                            f"switching to next model"
                        )
                        self._exhausted_models.add(current_model)
                        self._flush_model_status()
                        if not self._switch_to_next_model():
                            break

                    elif is_rate_limit:
                        retry_sec = self._parse_retry_seconds(err_str)
                        logger.debug(
                            f"[llm_client] rate-limit on {current_model} — "
                            f"retry_sec={retry_sec:.0f}"
                        )

                        # Gemini 429 with no parseable retry time = daily RPD exhausted.
                        # Old logic slept 65s and retried same model — causing infinite loop
                        # when quota is gone for the day. Now: switch immediately.
                        # If retry_sec IS parseable and short → genuine RPM, sleep briefly.
                        is_gemini_quota = (
                            _is_gemini_model(current_model)
                            and ("resource_exhausted" in err_str.lower()
                                 or "quota" in err_str.lower()
                                 or "429" in err_str)
                        )
                        if is_gemini_quota and retry_sec > _MAX_TPM_SLEEP_SEC:
                            # No parseable retry time → daily quota exhausted → switch NOW
                            logger.warning(
                                f"[llm_client] Gemini daily quota exhausted on {current_model} "
                                f"— switching to next model immediately"
                            )
                            self._exhausted_models.add(current_model)
                            self._tpd_exhausted.add(current_model)
                            self._flush_model_status()
                            if not self._switch_to_next_model():
                                break
                        elif is_gemini_quota and retry_sec <= _MAX_TPM_SLEEP_SEC:
                            # Short parseable wait → genuine RPM limit → sleep and retry
                            logger.warning(
                                f"[llm_client] Gemini RPM limit on {current_model} — "
                                f"sleeping {retry_sec+5:.0f}s then retrying"
                            )
                            time.sleep(retry_sec + 5)
                        elif retry_sec <= _MAX_TPM_SLEEP_SEC:
                            # TPM rate limit — switch to next model immediately instead
                            # of sleeping. After all models tried, smart recovery sleep
                            # waits only until the nearest retry_after window clears
                            # (typically 60-120s), far better than sleeping per-model.
                            recover_at = (
                                datetime.datetime.utcnow()
                                + datetime.timedelta(seconds=retry_sec)
                            ).isoformat() + "Z"
                            self._exhausted_models.add(current_model)
                            self._retry_after[current_model] = recover_at
                            self._flush_model_status()
                            logger.warning(
                                f"[llm_client] TPM limit on {current_model} "
                                f"(retry in {retry_sec:.0f}s) — switching to next model"
                            )
                            if not self._switch_to_next_model():
                                break
                        else:
                            # TPD exhausted — mark and switch
                            self._exhausted_models.add(current_model)
                            self._tpd_exhausted.add(current_model)
                            recover_at = (
                                datetime.datetime.utcnow()
                                + datetime.timedelta(seconds=retry_sec)
                            ).isoformat() + "Z"
                            self._retry_after[current_model] = recover_at
                            self._flush_model_status()
                            logger.warning(
                                f"[llm_client] TPD exhausted on {current_model} "
                                f"(recovers ~{recover_at}) — switching"
                            )
                            if not self._switch_to_next_model():
                                break

                    elif is_unavailable:
                        self._exhausted_models.add(current_model)
                        self._unavailable.add(current_model)
                        self._flush_model_status()
                        logger.warning(
                            f"[llm_client] {current_model} unavailable — switching"
                        )
                        if not self._switch_to_next_model():
                            break
                    else:
                        raise
            else:
                break

            # All models exhausted — wait until the nearest retry_after time,
            # or _RECOVERY_WAIT_SEC (whichever is shorter).
            # Smart sleep: if any model has a known retry_after, sleep until that
            # time (± 10s buffer) rather than a full 30-minute blanket wait.
            now_utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
            wait_sec = float(_RECOVERY_WAIT_SEC)
            if self._retry_after:
                nearest_secs = []
                for ra_str in self._retry_after.values():
                    try:
                        ra_dt = datetime.datetime.fromisoformat(
                            ra_str.replace("Z", "+00:00")
                        )
                        secs = max(0.0, (ra_dt - now_utc).total_seconds()) + 15
                        nearest_secs.append(secs)
                    except Exception:
                        pass
                if nearest_secs:
                    wait_sec = min(wait_sec, min(nearest_secs))
            wait_sec = max(60.0, wait_sec)  # floor: never less than 60s
            logger.warning(
                f"[llm_client] All models exhausted (tried: {self._exhausted_models}). "
                f"Waiting {wait_sec:.0f}s (smart sleep; attempt "
                f"{recovery_attempt+1}/{_MAX_RECOVERY_ATTEMPTS})..."
            )
            time.sleep(wait_sec)
            # Clear all exhausted/unavailable sets — rolling windows reset
            self._exhausted_models.clear()
            self._tpd_exhausted.clear()
            self._unavailable.clear()
            self._retry_after.clear()
            self._flush_model_status()
            # Restart from beginning of chain
            first = self._build_fallback_chain()[0]
            self._switch_to_model(first)

        raise RuntimeError(
            f"[llm_client] All providers exhausted after {_MAX_RECOVERY_ATTEMPTS} "
            f"recovery attempts. Last tried: {self._exhausted_models}"
        )

    def _build_fallback_chain(self) -> List[str]:
        """Gemini first (primary), Groq as fallback."""
        if self._gemini_key:
            return list(_GEMINI_FALLBACK_ORDER) + list(_GROQ_FALLBACK_ORDER)
        return list(_GROQ_FALLBACK_ORDER)

    def _switch_to_model(self, model_name: str) -> bool:
        """Switch to a specific model. Returns True on success."""
        try:
            old_model = self.config.model_name
            self._llm = self._build_llm(model_name)
            self.config.model_name = model_name
            self._provider = "gemini" if _is_gemini_model(model_name) else "groq"
            self._limits = _ALL_MODEL_LIMITS.get(model_name, _DEFAULT_LIMITS)
            logger.info(
                f"[llm_client] ✓ Switched: {old_model} → {model_name} "
                f"[{self._provider}] ({self._limits['tpd']:,} TPD)"
            )
            self._flush_model_status()
            return True
        except Exception as e:
            logger.error(f"[llm_client] Failed to switch to {model_name}: {e}")
            self._exhausted_models.add(model_name)
            return False

    def _switch_to_next_model(self) -> bool:
        """Switch to the next non-exhausted model in the full fallback chain."""
        for model_name in self._build_fallback_chain():
            if model_name not in self._exhausted_models:
                if self._switch_to_model(model_name):
                    return True
        return False

    # ── Retry time parser ─────────────────────────────────────────────────────

    def _parse_retry_seconds(self, error_msg: str) -> float:
        """Parse Groq/Gemini retry-after strings → total seconds."""
        # "Xh Ym Zs"
        m = re.search(r"try again in (\d+)h(\d+)m(\d+(?:\.\d+)?)s", error_msg)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        # "Xh Ym"
        m = re.search(r"try again in (\d+)h(\d+)m\b", error_msg)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60
        # "Xh"
        m = re.search(r"try again in (\d+)h\b", error_msg)
        if m:
            return int(m.group(1)) * 3600
        # "Xm Y.Zs"
        m = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", error_msg)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        # "X.Ys"
        m = re.search(r"try again in (\d+(?:\.\d+)?)s", error_msg)
        if m:
            return float(m.group(1))
        # Gemini quota errors rarely include retry time — treat as TPD
        logger.debug(f"[llm_client] Cannot parse retry time: {error_msg[:200]}")
        return _MAX_TPM_SLEEP_SEC + 1

    # ── Token tracking ────────────────────────────────────────────────────────

    def _record_usage(self, agent_name: str, response) -> None:
        try:
            tokens = 0
            meta = getattr(response, "usage_metadata", None) or {}
            tokens = int(meta.get("total_tokens", 0) or 0)
            if not tokens:
                rmeta = getattr(response, "response_metadata", None) or {}
                usage = rmeta.get("token_usage", {})
                tokens = int(usage.get("total_tokens", 0) or 0)

            with self._usage_lock:
                today = datetime.date.today().isoformat()
                if today != self._usage_date:
                    self._per_agent    = {}
                    self._total_tokens = 0
                    self._usage_date   = today
                entry = self._per_agent.setdefault(agent_name, {"tokens": 0, "calls": 0})
                entry["tokens"] += tokens
                entry["calls"]  += 1
                self._total_tokens += tokens

            self._flush_usage()
        except Exception as exc:
            logger.debug(f"[llm_client] token tracking error (non-fatal): {exc}")

    def _flush_usage(self) -> None:
        try:
            tpd = self._limits["tpd"]
            tpm = self._limits["tpm"]
            with self._usage_lock:
                data = {
                    "model":              self.config.model_name,
                    "provider":           self._provider,
                    "date":               self._usage_date,
                    "daily_limit_tokens": tpd,
                    "tpm_limit":          tpm,
                    "total_tokens_used":  self._total_tokens,
                    "tokens_remaining":   max(0, tpd - self._total_tokens),
                    "pct_used":           round(self._total_tokens / tpd, 4) if tpd else 0,
                    "per_agent":          dict(self._per_agent),
                    "last_updated":       datetime.datetime.utcnow().isoformat() + "Z",
                }
            self._token_file.parent.mkdir(parents=True, exist_ok=True)
            self._token_file.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug(f"[llm_client] token_usage flush error (non-fatal): {exc}")

    def _flush_model_status(self) -> None:
        try:
            data = {
                "active":        self.config.model_name,
                "provider":      self._provider,
                "all_models":    self._build_fallback_chain(),
                "groq_models":   _GROQ_FALLBACK_ORDER,
                "gemini_models": _GEMINI_FALLBACK_ORDER if self._gemini_key else [],
                "tpd_exhausted": sorted(self._tpd_exhausted),
                "unavailable":   sorted(self._unavailable),
                "retry_after":   dict(self._retry_after),
                "gemini_enabled": bool(self._gemini_key),
                "last_updated":  datetime.datetime.utcnow().isoformat() + "Z",
            }
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            self._status_file.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug(f"[llm_client] model_status flush error (non-fatal): {exc}")
