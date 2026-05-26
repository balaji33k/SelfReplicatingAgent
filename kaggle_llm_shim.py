"""
kaggle_llm_shim.py — Ollama-compatible OpenAI API shim for Kaggle

Loads a HuggingFace model (Qwen2.5-Coder-14B-Instruct) with 4-bit
quantization and exposes two endpoints that llm_client.py expects:

  GET  /api/tags              → tells llm_client Ollama is reachable
  POST /v1/chat/completions   → OpenAI-compatible inference

Run with:
    python kaggle_llm_shim.py --model-path /kaggle/input/... --port 11434
"""

import argparse
import json
import logging
import os
import time
import threading
import uuid
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("shim")

app = FastAPI(title="Kaggle LLM Shim")

# ── Global model state ────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_model_name = "qwen2.5-coder:14b"  # what llm_client sees


def load_model(model_path: str) -> None:
    global _model, _tokenizer

    log.info(f"Loading tokenizer from {model_path}")
    _tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    log.info("Loading model with 4-bit quantization (bitsandbytes)...")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_cfg,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    _model.eval()

    # VRAM used
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    log.info(f"✅ Model loaded — VRAM used: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen2.5-coder:14b"
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/tags")
def api_tags():
    """Ollama /api/tags — tells llm_client the model is available."""
    return {
        "models": [
            {
                "name": _model_name,
                "model": _model_name,
                "size": 9_000_000_000,
                "digest": "kaggle-local",
                "details": {"family": "qwen2.5-coder"},
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    if _model is None or _tokenizer is None:
        return JSONResponse({"error": "Model not loaded yet"}, status_code=503)

    # Apply chat template
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens or 4096,
            temperature=req.temperature or 0.2,
            do_sample=(req.temperature or 0.2) > 0,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    elapsed = time.time() - t0
    new_tokens = output_ids[0][input_len:]
    response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    n_tokens = len(new_tokens)

    log.info(f"Generated {n_tokens} tokens in {elapsed:.1f}s ({n_tokens/elapsed:.1f} tok/s)")

    # OpenAI response format
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": n_tokens,
            "total_tokens": input_len + n_tokens,
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to HF model dir")
    parser.add_argument("--model-name", default="qwen2.5-coder:14b")
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args()

    _model_name = args.model_name
    load_model(args.model_path)

    log.info(f"Starting shim on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
