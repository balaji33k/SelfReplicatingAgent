"""
llm_client.py — Multi-provider LLM client. All providers 100% free forever.

Model: Llama-4-Scout used across ALL providers (same model everywhere).
       Gemini kept as last resort (Google-only, different ecosystem).

Priority (max speed + least limited combined):
  1. Cerebras    — 2000 tok/s, least limited,  llama-4-scout ✅ fastest
  2. SambaNova   — 1500 tok/s, generous,       llama-4-scout ✅ same model
  3. Groq        —  800 tok/s, rolling TPM,    llama-4-scout ✅ same model
  4. OpenRouter  —  200 tok/s, soft limits,    llama-4-scout ✅ same model
  5. Gemini      —  400 tok/s, most limited,   gemini-2.0-flash (different ecosystem)
  6. Ollama      —   ~2 tok/s CPU,             qwen3:14b Q4  (emergency, 32 GB CPU only)

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
_OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "qwen3:14b")  # Q4 fits 32 GB CPU (~9 GB); Scout needs ~60 GB
_OLLAMA_CALL_TIMEOUT = 900  # CPU at ~2 tok/s — allow 15 min for a ~1000-token response on loaded system

# ── UNIVERSAL MODEL: Llama-4-Scout (same on all providers) ───────────────────
# Same model, same weights, consistent results across all free providers.
_SCOUT = "llama-4-scout"   # logical name — each provider has slightly different string

# ── Provider 1: Cerebras — least limited, 2000 tok/s ─────────────────────────
_CEREBRAS_API_KEY    = os.getenv("CEREBRAS_API_KEY", "")
_CEREBRAS_BASE_URL   = "https://api.cerebras.ai/v1"
_CEREBRAS_MODELS     = [
    "llama3.1-8b",   # Cerebras confirmed model (Llama-4-Scout not available on Cerebras)
]

# ── Provider 2: SambaNova — 1500 tok/s ───────────────────────────────────────
_SAMBANOVA_API_KEY   = os.getenv("SAMBANOVA_API_KEY", "")
_SAMBANOVA_BASE_URL  = "https://api.sambanova.ai/v1"
_SAMBANOVA_MODELS    = [
    "Meta-Llama-3.3-70B-Instruct",     # Llama-4-Scout deprecated on SambaNova; 70B is current best
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
    if ":" in model:                    return "ollama"      # Ollama model:tag (e.g. qwen3:14b)
    if model in _CEREBRAS_MODEL_SET:    return "cerebras"
    if model in _SAMBANOVA_MODEL_SET:   return "sambanova"
    if model in _OPENROUTER_MODEL_SET:  return "openrouter"
    if model in _GEMINI_MODEL_SET:      return "gemini"
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
    Unified LLM client with multi-provider fallback chain.

    ON KAGGLE:     Kaggle GPU model (Qwen shim) → Cerebras → SambaNova → Groq → OpenRouter → Gemini
    NOT ON KAGGLE: Cerebras → SambaNova → Groq → OpenRouter → Gemini → Ollama (CPU last resort)

    Switches instantly on rate limit — never waits when another provider is available.
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

        self._call_timeout = 30  # overridden per-model when Ollama is active
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

        # Pick starting model — API providers first; Ollama only as last resort
        chain = self._build_fallback_chain()
        if not chain:
            raise EnvironmentError("No API keys set for any provider.")
        self.config.model_name = chain[0]
        self._provider = _provider_of(chain[0])
        if self._provider == "ollama":
            self._call_timeout = _OLLAMA_CALL_TIMEOUT

        self._llm = self._build_llm(self.config.model_name)

        # Log active providers
        active = []
        if _CEREBRAS_API_KEY:       active.append("Cerebras")
        if _SAMBANOVA_API_KEY:      active.append("SambaNova")
        if _GROQ_API_KEY:           active.append("Groq")
        if _OPENROUTER_API_KEY:     active.append("OpenRouter")
        if _GEMINI_API_KEY:         active.append("Gemini")
        if self._use_ollama:        active.append(f"Ollama({self._ollama_model})🐢")
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

    @staticmethod
    def _on_kaggle() -> bool:
        """Detect Kaggle runtime via auto-set environment variables."""
        import os
        return bool(
            os.environ.get("KAGGLE_DATA_PROXY_TOKEN")
            or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        )

    @staticmethod
    def _cloud_fallback_allowed() -> bool:
        """Cloud fallback only if user explicitly sets ALLOW_CLOUD_FALLBACK=1."""
        import os
        return os.environ.get("ALLOW_CLOUD_FALLBACK", "").strip() == "1"

    def _build_fallback_chain(self) -> List[str]:
        """
        ON KAGGLE (auto-detected via KAGGLE_DATA_PROXY_TOKEN / KAGGLE_KERNEL_RUN_TYPE):
          - Only the Kaggle GPU model is in the chain.
          - If it fails, the run STOPS with a clear message asking the user to decide.
          - Cloud APIs are NOT added automatically — set ALLOW_CLOUD_FALLBACK=1 to opt in.

        ON KAGGLE with ALLOW_CLOUD_FALLBACK=1:
          Kaggle GPU → Cerebras → SambaNova → Groq → OpenRouter → Gemini

        NOT ON KAGGLE (cloud-only):
          Cerebras → SambaNova → Groq → OpenRouter → Gemini → Ollama (CPU last resort)
        """
        chain = []
        if self._on_kaggle():
            chain.append(self._ollama_model)            # Kaggle GPU model — only provider
            if self._cloud_fallback_allowed():          # opt-in cloud fallback
                if _CEREBRAS_API_KEY:   chain.extend(_CEREBRAS_MODELS)
                if _SAMBANOVA_API_KEY:  chain.extend(_SAMBANOVA_MODELS)
                if _GROQ_API_KEY:       chain.extend(_GROQ_MODELS)
                if _OPENROUTER_API_KEY: chain.extend(_OPENROUTER_MODELS)
                if _GEMINI_API_KEY:     chain.extend(_GEMINI_MODELS)
        else:
            if _CEREBRAS_API_KEY:   chain.extend(_CEREBRAS_MODELS)
            if _SAMBANOVA_API_KEY:  chain.extend(_SAMBANOVA_MODELS)
            if _GROQ_API_KEY:       chain.extend(_GROQ_MODELS)
            if _OPENROUTER_API_KEY: chain.extend(_OPENROUTER_MODELS)
            if _GEMINI_API_KEY:     chain.extend(_GEMINI_MODELS)
            if self._use_ollama:    chain.append(self._ollama_model)  # CPU last resort
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
        # Single unified chain: Cerebras → SambaNova → Groq → OpenRouter → Gemini → Ollama
        all_models = self._build_fallback_chain()

        for recovery_attempt in range(_MAX_RECOVERY_ATTEMPTS):
            for attempt in range(len(all_models) + 4):
                try:
                    # Inject /think or /no_think for Qwen3 models (Ollama CPU fallback)
                    msgs = lc_messages
                    if (self._provider == "ollama"
                            and self._ollama_model.startswith("qwen3")):
                        msgs = self._qwen3_inject_mode(lc_messages, agent_name)

                    is_qwen3_ollama = (self._provider == "ollama"
                                       and self._ollama_model.startswith("qwen3"))

                    def _call(llm, m, strip_think):
                        resp = llm.invoke(m)
                        content = resp.content
                        if strip_think:
                            content = re.sub(
                                r'<think>.*?</think>', '', content, flags=re.DOTALL
                            ).strip()
                        return content, resp

                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    fut = ex.submit(_call, self._llm, msgs, is_qwen3_ollama)
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
                                  or "not available" in err.lower()
                                  or "decommissioned" in err.lower()
                                  or "deprecat" in err.lower()))

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

            # All models exhausted
            if self._on_kaggle() and not self._cloud_fallback_allowed():
                # On Kaggle with no cloud fallback opted in — stop and tell the user.
                raise RuntimeError(
                    "\n"
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    "║  KAGGLE GPU MODEL FAILED — run halted.                       ║\n"
                    "║                                                              ║\n"
                    "║  The Kaggle GPU model (Qwen2.5-Coder shim) stopped          ║\n"
                    "║  responding. Cloud APIs were NOT used automatically.         ║\n"
                    "║                                                              ║\n"
                    "║  To switch to cloud APIs (Groq / Cerebras etc.):            ║\n"
                    "║    1. Add your API key to Kaggle Secrets                     ║\n"
                    "║       (e.g. GROQ_API_KEY at console.groq.com/keys)          ║\n"
                    "║    2. In Cell 9, add:                                        ║\n"
                    "║       os.environ['ALLOW_CLOUD_FALLBACK'] = '1'              ║\n"
                    "║    3. Re-run from Cell 9 onwards                             ║\n"
                    "╚══════════════════════════════════════════════════════════════╝"
                )

            # Not on Kaggle — smart sleep until nearest cloud provider recovers
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
            self._call_timeout = _OLLAMA_CALL_TIMEOUT if self._provider == "ollama" else 30
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
        # NOTE: m.groups() is a 0-indexed tuple — use m[0], m[1], m[2], not m[1], m[2], m[3]
        for pat, fn in [
            (r"try again in (\d+)h(\d+)m(\d+(?:\.\d+)?)s",
             lambda m: int(m[0])*3600 + int(m[1])*60 + float(m[2])),
            (r"try again in (\d+)h(\d+)m\b",
             lambda m: int(m[0])*3600 + int(m[1])*60),
            (r"try again in (\d+)h\b",
             lambda m: int(m[0])*3600),
            (r"try again in (\d+)m(\d+(?:\.\d+)?)s",
             lambda m: int(m[0])*60 + float(m[1])),
            (r"try again in (\d+(?:\.\d+)?)s",
             lambda m: float(m[0])),
        ]:
            try:
                m = re.search(pat, err)
                if m: return fn(m.groups())
            except (ValueError, IndexError):
                continue
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

            provider_labels = {
                "cerebras":  (1, "Cerebras   — 2000 tok/s, least limited ✅"),
                "sambanova": (2, "SambaNova  — 1500 tok/s, generous ✅"),
                "groq":      (3, "Groq       —  800 tok/s, rolling TPM ✅"),
                "openrouter":(4, "OpenRouter —  200 tok/s, soft limits ✅"),
                "gemini":    (5, "Gemini     —  400 tok/s, most limited ✅"),
                "ollama":    (6, "Ollama     —    ~2 tok/s, CPU emergency 🐢"),
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
