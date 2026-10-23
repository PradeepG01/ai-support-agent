# AI Support Agent

A production-style **LLM customer-support agent** for a hotel front desk, served as a **FastAPI** REST API.
It answers guest questions from a knowledge base (**RAG**), calls **tools** to look up bookings, open tickets
and escalate to humans, receives messages through a **signed webhook**, and ships with an **evaluation suite**
that blocks merges when answer quality drops.

![CI](https://github.com/PradeepG01/ai-support-agent/actions/workflows/ci.yml/badge.svg)

## What it does

```
guest message ──► POST /v1/chat  or  POST /v1/webhooks/messages (HMAC-signed, idempotent)
                     │
                     ▼
               SupportAgent.run()
                 1. retrieve top-k knowledge chunks (TF-IDF RAG)
                 2. build system prompt (versioned) + conversation history
                 3. LLM call ──► tool calls? ──► execute tool ──► feed result back ──► repeat (max N steps)
                 4. final grounded reply, persisted with sources, latency, prompt version
                     │
                     ▼
     PostgreSQL / SQLite: conversations, messages, bookings, tickets, webhook events
```

| Capability | Where |
|---|---|
| Retrieval-augmented generation (chunking + TF-IDF ranking, swappable for a vector DB) | `app/retrieval.py` |
| Tool / function calling: `lookup_booking`, `create_ticket`, `escalate_to_human` with JSON schemas | `app/tools.py` |
| Provider-agnostic LLM layer: OpenAI, Anthropic, and an offline mock, with retries + backoff | `app/llm.py` |
| Agent loop with step budget, tool-error recovery and auto-escalation guardrails | `app/agent.py` |
| Versioned prompt templates | `app/prompts.py` |
| Inbound webhook with HMAC-SHA256 signature, replay window and idempotency | `app/webhooks.py`, `app/main.py` |
| Structured JSON logs + Prometheus-style `/metrics` | `app/logging_config.py`, `app/main.py` |
| LLM document extraction: guest email → schema-validated booking JSON, with one self-correction retry | `app/extraction.py` |
| Data migration: import bookings from CSV/Excel with row-level validation and idempotent upserts | `scripts/import_bookings.py` |
| Offline eval harness (tool routing, answer accuracy, retrieval hit rate) | `evals/` |
| Docker, docker-compose (API + PostgreSQL), GitHub Actions CI, Kubernetes manifests | repo root, `k8s/` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload            # runs offline with LLM_PROVIDER=mock
```

```bash
curl -X POST localhost:8000/v1/chat -H 'X-API-Key: dev-api-key' -H 'Content-Type: application/json' \
  -d '{"customer_ref": "guest-1", "message": "Can you check my booking SG1234?"}'
```

Use a real model:

```bash
export LLM_PROVIDER=openai OPENAI_API_KEY=sk-... LLM_MODEL=gpt-4o-mini
# or
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... LLM_MODEL=<anthropic model id>
```

With Docker + PostgreSQL: `docker compose up --build`

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat` | Send a guest message, get the agent's reply, tool calls and sources |
| `GET` | `/v1/conversations/{id}` | Full transcript, including tool results and metadata |
| `POST` | `/v1/knowledge` | Ingest or replace a knowledge-base document |
| `GET` | `/v1/knowledge` | List documents |
| `POST` | `/v1/webhooks/messages` | Inbound channel webhook (WhatsApp/SMS provider), HMAC-signed |
| `POST` | `/v1/extract/booking-request` | Extract a structured, validated booking request from a free-text email |
| `GET` | `/health`, `/metrics` | Health check and metrics |

Interactive docs: `http://localhost:8000/docs`

## Tests and evals

```bash
pytest -q                          # 24 unit + API tests
python -m evals.run_evals          # 18 end-to-end agent scenarios
python -m scripts.import_bookings bookings.xlsx --dry-run   # validate a customer's spreadsheet before importing
```

Current eval result with the offline model: **18/18 pass** (tool routing 18/18, answer accuracy 18/18, retrieval 11/11).
Run the same suite with `LLM_PROVIDER=openai` or `anthropic` to compare models and prompts.

## Design notes

* **Guardrails over trust:** the agent has a hard step budget; unknown tools and bad arguments are returned to the
  model as errors so it can self-correct; exhausting the budget escalates to a human instead of looping.
* **Grounding:** the system prompt restricts answers to retrieved context and tool results; the eval suite checks it.
* **Webhooks are retried by providers**, so events are de-duplicated by `event_id` and the original reply is replayed.
* **Swappable parts:** retrieval, LLM provider and storage are each behind a small interface.
