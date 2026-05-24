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

A self-replicating AI coding agent that evolves across generations using Groq/Llama-3.3.

## Architecture

- **7-agent pipeline**: Analyst → Architect → Coder → Critic → Reviser → Executor → Debugger
- **Groq / Llama-3.3-70b** backend — fast, free inference
- **Live MISSION_CONTROL dashboard** — polls every 3 seconds
- **Clone Inspector** — semantic originality check on each spawned generation
- **Per-file evolution** — stays within Groq free-tier token budget

## Dashboard

Visit `/MISSION_CONTROL.html` after deployment to watch the live evolution.

## Setup

Set `GROQ_API_KEY` as a Space secret in Settings → Variables and secrets.
