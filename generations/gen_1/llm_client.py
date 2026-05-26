"""
llm_client.py — Multi-provider LLM client. All providers 100% free forever.

Model: Llama-4-Scout used across ALL providers (same model everywhere).
       Gemini kept as last resort (Google-only, different ecosystem).

Priority (least limited → most limited):
  0. Ollama      — local GPU, unlimited,       llama4:scout (or qwen3:14b)
  1. Cerebras    — least limited, 2000 tok/s,  llama-4-scout ✅ same model
  2. SambaNova   — 1500 tok/s,                 llama-4-scout ✅ same model
  3. OpenRouter  — free :free tier,            llama-4-scout ✅ same model
  4. Groq        — rolling TPM,                llama-4-scout ✅ same model
  5. Gemini      — last resort, 1500 RPD       gemini-2.0-flash (fallback only)

Why Llama-4-Scout:
  - Only model available FREE on Cerebras + SambaNova + OpenRouter + Groq
  - 17B active params (MoE, 109B total) — fast and capable
  - Released April 2025 by Meta
  - Consistent results across all providers (same weights)

Combined free pool: ~5M+ tokens/day. Effectively unlimited.

Env vars (all free, no credit card needed):
  OLLAMA_BASE_URL    — http://localhost:11434  (Colab/Kaggle)
  OLLAMA_MODEL       — default: llama4:scout
  CEREBRAS_API_KEY   — free at cerebras.ai
  SAMBANOVA_API_KEY  — free at sambanova.ai
  OPENROUTER_API_KEY — free at openrouter.ai
  GROQ_API_KEY       — free at console.groq.com
  GEMINI_API_KEY     — free at aistudio.google.com (last resort)
"""
import concurrent.futures
import datetime
import json
import logging
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ── Provider 0: Ollama (local, no limits) ────────────────────────────────────
_OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "")
_OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "llama4:scout")  # same model as all API providers
_OLLAMA_CALL_TIMEOUT = 180

# ── UNIVERSAL MODEL: Llama-4-Scout (same on all providers) ───────────────────
# Same model, same weights, consistent results across all free providers.
_SCOUT = "llama-4-scout"   # logical name — each provider has slightly different string

# ── Provider 1: Cerebras — least limited, 2000 tok/s ─────────────────────────
_CEREBRAS_API_KEY    = os.getenv("CEREBRAS_API_KEY", "")
_CEREBRAS_BASE_URL   = "https://api.cerebras.ai/v1"
_CEREBRAS_MODELS     = [
    "llama-4-scout-17b-16e-instruct",  # ✅ Llama-4-Scout only
]

# ── Provider 2: SambaNova — 1500 tok/s ───────────────────────────────────────
_SAMBANOVA_API_KEY   = os.getenv("SAMBANOVA_API_KEY", "")
_SAMBANOVA_BASE_URL  = "https://api.sambanova.ai/v1"
_SAMBANOVA_MODELS    = [
    "Llama-4-Scout-17B-16E-Instruct",  # ✅ Llama-4-Scout only
]

# ── Provider 3: OpenRouter — free :free tier ──────────────────────────────────
_OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_MODELS   = [
    "meta-llama/llama-4-scout:free",   # ✅ Llama-4-Scout only
]

# ── Provider 4: Groq — rolling TPM limits ────────────────────────────────────
_GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
_GROQ_MODELS         = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # ✅ Llama-4-Scout only
]

# ── Provider 5: Gemini — last resort (Google-only, different ecosystem) ───────
_GEMINI_API_KEY      = (
    os.getenv("GEMINI_API_KEY", "")
    or os.getenv("GOOGLE_API_KEY", "")
    or os.getenv("GoogleAPIKey", "")
)
_GEMINI_MODELS       = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# ── Rate limit thresholds ─────────────────────────────────────────────────────
_MAX_TPM_SLEEP_SEC     = 900    # >15 min wait → treat as daily limit, switch model
_RECOVERY_WAIT_SEC     = 1800   # fallback wait if all providers exhausted (30 min)
_MAX_RECOVERY_ATTEMPTS = 4      # give up after 4 × 30 min = 2 hrs
_GEMINI_RPM_SLEEP_SEC  = 65     # 60s RPM window + 5s buffer

# Models that belong to each provider (for routing)
_CEREBRAS_MODEL_SET  = set(_CEREBRAS_MODELS)
_SAMBANOVA_MODEL_SET = set(_SAMBANOVA_MODELS)
_OPENROUTER_MODEL_SET= set(_OPENROUTER_MODELS)
_GEMINI_MODEL_SET    = set(_GEMINI_MODELS)


def _provider_of(model: str) -> str:
    if model in _CEREBRAS_MODEL_SET:  return "cerebras"
    if model in _SAMBANOVA_MODEL_SET: return "sambanova"
    if model in _OPENROUTER_MODEL_SET:return "openrouter"
    if model in _GEMINI_MODEL_SET:    return "gemini"
    return "groq"


def _check_ollama_reachable() -> bool:
    if not _OLLAMA_BASE_URL:
        return False
    try:
        urllib.request.urlopen(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


class LLMClient:
    """
    Unified LLM client with 7-provider fallback chain.
    Priority: Ollama → Cerebras → SambaNova → Together → Groq → OpenRouter → Gemini
    Least limited first — switches instantly on rate limit, never waits when
    another provider is available.
    """

    _usage_lock = threading.Lock()

    def __init__(self, config):
        self.config = config

        # Detect available providers
        self._use_ollama  = bool(_OLLAMA_BASE_URL) and _check_ollama_reachable()
        self._ollama_url  = _OLLAMA_BASE_URL
        self._ollama_model= _OLLAMA_MODEL

        # At least one provider must be available
        has_any = (
            self._use_ollama
            or _CEREBRAS_API_KEY
            or _SAMBANOVA_API_KEY
            or _GROQ_API_KEY
            or _OPENROUTER_API_KEY
            or _GEMINI_API_KEY
        )
        if not has_any:
            raise EnvironmentError(
                "No LLM provider available. Set at least one of: "
                "OLLAMA_BASE_URL, CEREBRAS_API_KEY, SAMBANOVA_API_KEY, "
                "GROQ_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY"
            )

        self._call_timeout = _OLLAMA_CALL_TIMEOUT if self._use_ollama else 30
        self._exhausted_models: set = set()
        self._tpd_exhausted:    set = set()
        self._unavailable:      set = set()
        self._retry_after: Dict[str, str] = {}

        # Token tracking
        self._usage_date   = datetime.date.today().isoformat()
        self._per_agent: Dict = {}
        self._total_tokens   = 0

        # Resolve project root for data files
        gen_dir = Path(__file__).resolve().parent
        project_root = (
            gen_dir.parent.parent
            if gen_dir.parent.name == "generations"
            else gen_dir.parent
        )
        self._token_file  = project_root / "data" / "token_usage.json"
        self._status_file = project_root / "data" / "model_status.json"

        # Pick starting model and build LLM
        if self._use_ollama:
            self._provider    = "ollama"
            self.config.model_name = self._ollama_model
        else:
            chain = self._build_fallback_chain()
            if not chain:
                raise EnvironmentError("No API keys set for any provider.")
            self.config.model_name = chain[0]
            self._provider = _provider_of(chain[0])

        self._llm = self._build_llm(self.config.model_name)

        # Log active providers
        active = []
        if self._use_ollama:        active.append(f"Ollama({self._ollama_model})")
        if _CEREBRAS_API_KEY:       active.append("Cerebras")
        if _SAMBANOVA_API_KEY:      active.append("SambaNova")
        if _OPENROUTER_API_KEY:     active.append("OpenRouter")
        if _GROQ_API_KEY:           active.append("Groq")
        if _GEMINI_API_KEY:         active.append("Gemini")
        logger.info(
            f"[llm_client] Active providers: {' → '.join(active)}\n"
            f"             Starting model: {self.config.model_name}"
        )
        self._flush_model_status()

    # ── LLM factory ──────────────────────────────────────────────────────────

    def _build_llm(self, model_name: str):
        """Build the right LangChain LLM object for the given model."""
        from langchain_openai import ChatOpenAI

        # Ollama
        if self._use_ollama and model_name == self._ollama_model:
            return ChatOpenAI(
                model=self._ollama_model,
                base_url=f"{self._ollama_url}/v1",
                api_key="ollama",
                temperature=self.config.temperature,
                max_tokens=8192,
                timeout=_OLLAMA_CALL_TIMEOUT,
                max_retries=1,
            )

        provider = _provider_of(model_name)

        # Cerebras
        if provider == "cerebras":
            if not _CEREBRAS_API_KEY:
                raise EnvironmentError("CEREBRAS_API_KEY not set")
            return ChatOpenAI(
                model=model_name,
                base_url=_CEREBRAS_BASE_URL,
                api_key=_CEREBRAS_API_KEY,
                temperature=self.config.temperature,
                max_tokens=8192,
                timeout=60,
                max_retries=1,
            )

        # SambaNova
        if provider == "sambanova":
            if not _SAMBANOVA_API_KEY:
                raise EnvironmentError("SAMBANOVA_API_KEY not set")
            return ChatOpenAI(
                model=model_name,
                base_url=_SAMBANOVA_BASE_URL,
                api_key=_SAMBANOVA_API_KEY,
                temperature=self.config.temperature,
                max_tokens=8192,
                timeout=60,
                max_retries=1,
            )

        # OpenRouter
        if provider == "openrouter":
            if not _OPENROUTER_API_KEY:
                raise EnvironmentError("OPENROUTER_API_KEY not set")
            return ChatOpenAI(
                model=model_name,
                base_url=_OPENROUTER_BASE_URL,
                api_key=_OPENROUTER_API_KEY,
                temperature=self.config.temperature,
                max_tokens=8192,
                timeout=90,
                max_retries=1,
                default_headers={
                    "HTTP-Referer": "https://huggingface.co/spaces/Balaji33k/self-replicating-agent",
                    "X-Title": "SelfReplicatingAgent",
                },
            )

        # Gemini
        if provider == "gemini":
            if not _GEMINI_API_KEY:
                raise EnvironmentError("GEMINI_API_KEY not set")
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=_GEMINI_API_KEY,
                temperature=self.config.temperature,
                max_output_tokens=8192,
                timeout=90,
                max_retries=1,
                transport="rest",
            )

        # Groq (default)
        if not _GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not set")
        return ChatGroq(
            model=model_name,
            temperature=self.config.temperature,
            api_key=_GROQ_API_KEY,
            max_tokens=8192,
            timeout=90,
            max_retries=1,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def call(self, prompt: str, agent_name: str = "") -> str:
        return self._invoke_with_retry([HumanMessage(content=prompt)], agent_name or "unknown")

    def chat_completion(self, messages: List[Dict],
                        response_format: Optional[Dict] = None,
                        agent_name: str = "") -> str:
        lc_messages = []
        for msg in messages:
            role, content = msg.get("role", "user"), msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        return self._invoke_with_retry(lc_messages, agent_name or "unknown")

    # ── Fallback chain ────────────────────────────────────────────────────────

    def _build_fallback_chain(self) -> List[str]:
        """
        All providers 100% free forever. Ordered: least limited → most limited.

          1. Cerebras    — least limited, 2000 tok/s (Llama)
          2. SambaNova   — 1500 tok/s (Llama)
          3. OpenRouter  — free Qwen3-14B:free
          4. Groq        — free Qwen-QwQ-32B, rolling TPM
          5. Gemini      — last resort, 1500 RPD

        Ollama handled as fast-path before this chain (if running locally).
        """
        chain = []
        if _CEREBRAS_API_KEY:   chain.extend(_CEREBRAS_MODELS)    # least limited
        if _SAMBANOVA_API_KEY:  chain.extend(_SAMBANOVA_MODELS)   # 2nd least
        if _OPENROUTER_API_KEY: chain.extend(_OPENROUTER_MODELS)  # free Qwen3
        if _GROQ_API_KEY:       chain.extend(_GROQ_MODELS)        # free Qwen
        if _GEMINI_API_KEY:     chain.extend(_GEMINI_MODELS)      # last resort
        return chain

    # ── Qwen3 thinking mode ───────────────────────────────────────────────────

    def _qwen3_inject_mode(self, lc_messages, agent_name: str):
        """Inject /think or /no_think prefix for Qwen3 models."""
        _THINK_AGENTS = {"analyst", "architect", "meta_architect", "evolution_engine"}
        use_think = agent_name.lower() in _THINK_AGENTS
        prefix = "/think\n" if use_think else "/no_think\n"
        patched = []
        for msg in lc_messages:
            if isinstance(msg, HumanMessage) and not msg.content.startswith("/think"):
                patched.append(HumanMessage(content=prefix + msg.content))
            else:
                patched.append(msg)
        return patched

    # ── Core retry / fallback logic ───────────────────────────────────────────

    def _invoke_with_retry(self, lc_messages, agent_name: str) -> str:

        # ── Ollama fast-path ──────────────────────────────────────────────────
        if self._use_ollama:
            if self._ollama_model.startswith("qwen3"):
                lc_messages = self._qwen3_inject_mode(lc_messages, agent_name)
            for attempt in range(3):
                try:
                    def _call(llm, msgs):
                        resp = llm.invoke(msgs)
                        content = re.sub(r'<think>.*?</think>', '',
                                         resp.content, flags=re.DOTALL).strip()
                        return content, resp
                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    fut = ex.submit(_call, self._llm, lc_messages)
                    try:
                        content, resp = fut.result(timeout=self._call_timeout)
                        ex.shutdown(wait=False)
                    except concurrent.futures.TimeoutError:
                        ex.shutdown(wait=False)
                        raise TimeoutError(f"Ollama timed out after {self._call_timeout}s")
                    try: self._record_usage(agent_name, resp)
                    except Exception: pass
                    return content
                except Exception as exc:
                    logger.warning(f"[llm_client] Ollama attempt {attempt+1}/3: {exc!s:.120}")
                    if attempt < 2: time.sleep(5)
                    else: raise RuntimeError(f"Ollama failed after 3 attempts: {exc}") from exc

        # ── API provider chain ────────────────────────────────────────────────
        all_models = self._build_fallback_chain()

        for recovery_attempt in range(_MAX_RECOVERY_ATTEMPTS):
            for attempt in range(len(all_models) + 4):
                try:
                    def _call(llm, msgs):
                        resp = llm.invoke(msgs)
                        return resp.content, resp

                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    fut = ex.submit(_call, self._llm, lc_messages)
                    try:
                        content, resp = fut.result(timeout=self._call_timeout)
                        ex.shutdown(wait=False)
                    except concurrent.futures.TimeoutError:
                        ex.shutdown(wait=False)
                        raise TimeoutError(f"LLM timed out after {self._call_timeout}s")
                    self._record_usage(agent_name, resp)
                    return content

                except Exception as exc:
                    err = str(exc)
                    is_rate   = ("429" in err or "rate_limit" in err
                                 or "RateLimitError" in type(exc).__name__
                                 or "quota" in err.lower()
                                 or "resource_exhausted" in err.lower())
                    is_timeout= ("timeout" in err.lower() or "timed out" in err.lower()
                                 or type(exc).__name__ in
                                    ("TimeoutError","ReadTimeout","ConnectTimeout"))
                    is_unavail= (not is_timeout and
                                 ("404" in err or "model_not_found" in err
                                  or "does not exist" in err
                                  or "not found" in err.lower()
                                  or "decommissioned" in err.lower()))

                    cur = self.config.model_name

                    if is_timeout or is_unavail:
                        reason = "timeout" if is_timeout else "unavailable"
                        logger.warning(f"[llm_client] {cur} {reason} — switching")
                        self._exhausted_models.add(cur)
                        if is_unavail: self._unavailable.add(cur)
                        self._flush_model_status()
                        if not self._switch_to_next_model(): break

                    elif is_rate:
                        retry_sec = self._parse_retry_seconds(err)
                        recover_at = (
                            datetime.datetime.utcnow()
                            + datetime.timedelta(seconds=retry_sec)
                        ).isoformat() + "Z"
                        self._exhausted_models.add(cur)
                        self._retry_after[cur] = recover_at
                        if retry_sec > _MAX_TPM_SLEEP_SEC:
                            self._tpd_exhausted.add(cur)
                        self._flush_model_status()
                        logger.warning(
                            f"[llm_client] {_provider_of(cur).upper()} rate-limit on "
                            f"{cur} (retry in {retry_sec:.0f}s) → switching"
                        )
                        if not self._switch_to_next_model(): break
                    else:
                        raise
            else:
                break

            # All models exhausted — smart sleep until nearest recovery
            now_utc  = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
            wait_sec = float(_RECOVERY_WAIT_SEC)
            if self._retry_after:
                secs = []
                for ra in self._retry_after.values():
                    try:
                        dt = datetime.datetime.fromisoformat(ra.replace("Z", "+00:00"))
                        secs.append(max(0.0, (dt - now_utc).total_seconds()) + 15)
                    except Exception:
                        pass
                if secs:
                    wait_sec = min(wait_sec, min(secs))
            wait_sec = max(60.0, wait_sec)
            logger.warning(
                f"[llm_client] All {len(self._exhausted_models)} models exhausted. "
                f"Sleeping {wait_sec:.0f}s then resetting "
                f"(attempt {recovery_attempt+1}/{_MAX_RECOVERY_ATTEMPTS})..."
            )
            time.sleep(wait_sec)
            self._exhausted_models.clear()
            self._tpd_exhausted.clear()
            self._unavailable.clear()
            self._retry_after.clear()
            self._flush_model_status()
            first = self._build_fallback_chain()[0]
            self._switch_to_model(first)

        raise RuntimeError(
            f"[llm_client] All providers exhausted after "
            f"{_MAX_RECOVERY_ATTEMPTS} recovery attempts."
        )

    def _switch_to_model(self, model_name: str) -> bool:
        try:
            old = self.config.model_name
            self._llm = self._build_llm(model_name)
            self.config.model_name = model_name
            self._provider = _provider_of(model_name)
            self._call_timeout = 30
            logger.info(f"[llm_client] ✓ {old} → {model_name} [{self._provider.upper()}]")
            self._flush_model_status()
            return True
        except Exception as e:
            logger.error(f"[llm_client] Cannot switch to {model_name}: {e}")
            self._exhausted_models.add(model_name)
            return False

    def _switch_to_next_model(self) -> bool:
        for m in self._build_fallback_chain():
            if m not in self._exhausted_models:
                return self._switch_to_model(m)
        return False

    # ── Retry time parser ─────────────────────────────────────────────────────

    def _parse_retry_seconds(self, err: str) -> float:
        for pat, fn in [
            (r"try again in (\d+)h(\d+)m(\d+(?:\.\d+)?)s",
             lambda m: int(m[1])*3600 + int(m[2])*60 + float(m[3])),
            (r"try again in (\d+)h(\d+)m\b",
             lambda m: int(m[1])*3600 + int(m[2])*60),
            (r"try again in (\d+)h\b",
             lambda m: int(m[1])*3600),
            (r"try again in (\d+)m(\d+(?:\.\d+)?)s",
             lambda m: int(m[1])*60 + float(m[2])),
            (r"try again in (\d+(?:\.\d+)?)s",
             lambda m: float(m[1])),
        ]:
            m = re.search(pat, err)
            if m: return fn(m.groups())
        return _MAX_TPM_SLEEP_SEC + 1

    # ── Token tracking ────────────────────────────────────────────────────────

    def _record_usage(self, agent_name: str, response) -> None:
        try:
            tokens = 0
            meta = getattr(response, "usage_metadata", None) or {}
            tokens = int(meta.get("total_tokens", 0) or 0)
            if not tokens:
                rmeta = getattr(response, "response_metadata", None) or {}
                tokens = int(rmeta.get("token_usage", {}).get("total_tokens", 0) or 0)
            with self._usage_lock:
                today = datetime.date.today().isoformat()
                if today != self._usage_date:
                    self._per_agent = {}; self._total_tokens = 0
                    self._usage_date = today
                e = self._per_agent.setdefault(agent_name, {"tokens": 0, "calls": 0})
                e["tokens"] += tokens
                e["calls"]  += 1
                self._total_tokens += tokens
            self._flush_token_usage()
        except Exception:
            pass

    def _flush_token_usage(self):
        try:
            self._token_file.parent.mkdir(exist_ok=True)
            data = {"date": self._usage_date, "total_tokens": self._total_tokens,
                    "per_agent": self._per_agent}
            self._token_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _flush_model_status(self):
        try:
            self._status_file.parent.mkdir(exist_ok=True)
            chain = self._build_fallback_chain()
            statuses = {}

            if self._use_ollama:
                statuses["ollama"] = {
                    "status": "active", "provider": "ollama",
                    "model": self._ollama_model, "priority": 0
                }

            provider_labels = {
                "cerebras":  (1, "Cerebras   — least limited, 2000 tok/s ✅ free"),
                "sambanova": (2, "SambaNova  — 1500 tok/s ✅ free"),
                "openrouter":(3, "OpenRouter — Qwen3-14B:free ✅ free"),
                "groq":      (4, "Groq       — Qwen-QwQ-32B ✅ free"),
                "gemini":    (5, "Gemini     — last resort ✅ free"),
            }

            for m in chain:
                p = _provider_of(m)
                pri, label = provider_labels.get(p, (9, p))
                if m in self._unavailable:   st = "unavailable"
                elif m in self._tpd_exhausted: st = "tpd_exhausted"
                elif m in self._exhausted_models: st = "exhausted"
                elif m == self.config.model_name: st = "active"
                else: st = "available"
                ra = self._retry_after.get(m)
                statuses[m] = {"status": st, "provider": p,
                               "provider_label": label, "priority": pri}
                if ra: statuses[m]["retry_after"] = ra

            self._status_file.write_text(json.dumps({
                "current_model": self.config.model_name,
                "current_provider": self._provider,
                "models": statuses,
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z"
            }, indent=2))
        except Exception:
            pass
