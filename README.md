---
title: Self Replicating Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Self-Replicating AI Agent

A self-replicating AI coding agent that evolves across generations.
Each generation benchmarks itself, analyzes failures, and spawns an improved next generation.

## Architecture

- **7-agent pipeline**: Analyst → Architect → Coder → Critic → Reviser → Executor → Debugger
- **Llama-4-Scout** across all providers — same model, consistent results
- **6-provider fallback chain** — auto-switches on rate limit, never stops
- **Live MISSION_CONTROL dashboard** — polls every 3 seconds
- **Clone Inspector** — semantic originality check on each spawned generation

## Provider Chain (fastest → slowest)

| Priority | Provider | Speed | Model |
|---|---|---|---|
| 1 | Cerebras | 2000 tok/s | llama-4-scout |
| 2 | SambaNova | 1500 tok/s | llama-4-scout |
| 3 | Groq | 800 tok/s | llama-4-scout |
| 4 | OpenRouter | 200 tok/s | llama-4-scout |
| 5 | Gemini | 400 tok/s | gemini-2.0-flash |
| 6 | Ollama (CPU) | ~2 tok/s | qwen3:14b (emergency) |

## Dashboard

Visit `/MISSION_CONTROL.html` after deployment to watch the live evolution.

## Setup — Add Secrets in Space Settings

| Secret | Where to get it |
|---|---|
| `GROQ_API_KEY` | console.groq.com (free) |
| `GEMINI_API_KEY` | aistudio.google.com (free) |
| `CEREBRAS_API_KEY` | cloud.cerebras.ai (free) |
| `SAMBANOVA_API_KEY` | cloud.sambanova.ai (free) |
| `OPENROUTER_API_KEY` | openrouter.ai (free) |

Minimum: set `GROQ_API_KEY` to start. More keys = more capacity + faster fallback.
