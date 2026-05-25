"""
llm_client.py — Single-provider LLM client for Groq.

Uses whatever model is set in LLMConfig (default: deepseek-r1-distill-llama-70b).
Retries with exponential backoff on transient errors.

Provider auto-detected from environment variables:
  GROQ_API_KEY   → Groq (primary)
  GEMINI_API_KEY → Gemini (if no Groq key)
  GOOGLE_GENAI_USE_VERTEXAI=TRUE + GCLOUD_ACCESS_TOKEN → Vertex AI
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 7
RETRY_BASE_WAIT = 5

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMClient:
    """
    LLM client for Groq (primary), Gemini, or Vertex AI.
    Provider is auto-detected from environment variables.
    Retries with exponential backoff on transient errors.
    """

    def __init__(self, config):
        self.config = config  # LLMConfig — model_name, temperature, timeout_seconds

    def _detect_provider(self):
        """Return ('groq', key) | ('gemini', key) | ('vertex', token)."""
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            return "groq", groq_key

        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE":
            token = os.getenv("GCLOUD_ACCESS_TOKEN")
            if not token:
                raise EnvironmentError("GCLOUD_ACCESS_TOKEN not set for Vertex AI mode.")
            return "vertex", token

        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            return "gemini", gemini_key

        raise EnvironmentError(
            "No API key found. Set GROQ_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY."
        )

    def _build_groq_request(self, prompt: str, api_key: str) -> urllib.request.Request:
        model = self.config.model_name
        # If config still has a Gemini model name, fall back to deepseek
        if "gemini" in model.lower():
            model = "deepseek-r1-distill-llama-70b"
            logger.info(f"[llm_client] Gemini model name in config; using {model} for Groq")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": 8192,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "python-groq/0.9.0",
        }
        return urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(payload).encode(),
            headers=headers,
        )

    def _build_gemini_request(self, prompt: str, api_key: str) -> urllib.request.Request:
        model = self.config.model_name
        if "gemini" not in model.lower():
            model = "gemini-2.0-flash"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        return urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers
        )

    def _build_vertex_request(self, prompt: str, token: str) -> urllib.request.Request:
        model = self.config.model_name
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "codingagentproject-487605")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{model}:generateContent"
        )
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        return urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers
        )

    def _parse_response(self, data: dict, provider: str) -> str:
        if provider == "groq":
            return data["choices"][0]["message"]["content"]
        else:  # gemini / vertex
            return data["candidates"][0]["content"]["parts"][0]["text"]

    def call(self, prompt: str) -> str:
        """
        Send a prompt and return the response text.
        Retries with exponential backoff on transient errors (up to MAX_RETRIES).
        """
        provider, credential = self._detect_provider()
        logger.debug(f"[llm_client] provider={provider} model={self.config.model_name}")

        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                if provider == "groq":
                    req = self._build_groq_request(prompt, credential)
                elif provider == "vertex":
                    req = self._build_vertex_request(prompt, credential)
                else:
                    req = self._build_gemini_request(prompt, credential)

                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode())
                    return self._parse_response(data, provider)

            except urllib.error.HTTPError as e:
                body = e.read().decode()
                last_error = f"HTTP {e.code} {e.reason} — {body[:300]}"
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                logger.error(
                    f"LLM call failed (attempt {attempt+1}/{MAX_RETRIES}): "
                    f"{last_error}. Retrying in {wait}s..."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)

            except Exception as e:
                last_error = str(e)
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                logger.error(
                    f"LLM call failed (attempt {attempt+1}/{MAX_RETRIES}): {last_error}. "
                    f"Retrying in {wait}s..."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} attempts. "
            f"Provider={provider}, model={self.config.model_name}. "
            f"Last error: {last_error}"
        )

    def chat_completion(
        self, messages: List[Dict], response_format: Optional[Dict] = None
    ) -> str:
        """
        Chat-style interface — flattens messages to a single prompt.
        Kept for backwards compatibility. `response_format` is ignored.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[Instructions]\n{content}")
            else:
                parts.append(content)
        return self.call("\n\n".join(parts))
