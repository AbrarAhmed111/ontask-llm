<div align="center">

<a href="https://www.abrarahmed.pro" target="_blank">
  <img src="https://www.abrarahmed.pro/assets/devAbby-fulllogo-C9-MX7QK.png" alt="Built by Abrar Ahmed" height="65" />
</a>

# 🧠 ontask-llm

**OnTask's AI microservice** — turns an already-computed structured snapshot of a workspace's
daily focus-time activity into a validated, grounded narrative, using a multi-provider LLM
Gateway (LangChain) with automatic failover.

Built by **[Abrar Ahmed](https://www.abrarahmed.pro)** | Managed with [uv](https://docs.astral.sh/uv/)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-core%20%2B%20openai-1C3C3C.svg)](https://python.langchain.com)
[![Package Manager](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

This service exists for exactly one feature in [OnTask](https://www.abrarahmed.pro): **Phase 10
— the Shared "Yesterday's Work" Summary**. A workspace's day of focus-time tracking is already
fully computed elsewhere (in the Next.js app, from `task_time_entries`) before it ever reaches
this service — this service's only job is to turn that structured dataset into a few sentences
of neutral, factual prose.

## 🧭 Design principle: structured-first, AI-narrates-only

The model is never the source of a fact. Every number, name, and status the user sees comes
from the caller's `StructuredSnapshot`; the AI's output (`SummaryNarrative`) is validated
against that snapshot before it's returned, and any hallucinated member id or untraceable
figure triggers a corrective retry — and, failing that, a deterministic, non-AI fallback
narrative, so this endpoint can never hand back a fabricated report.

```
Caller (Next.js) aggregates task_time_entries for one workspace + day
    ↓
POST /api/summary/generate  { snapshot: StructuredSnapshot }
    ↓
No activity that day? → deterministic sentence, LLM never called
    ↓ has activity
System prompt (hard rules) + snapshot JSON → LLMGateway.generate_structured(SummaryNarrative)
    ↓
Validate: every member_id is known · every number is traceable to the snapshot
    ↓ invalid                                   ↓ valid
Ask the model to correct itself, retry once     Return narrative + snapshot + generation metadata
    ↓ still invalid
Deterministic template narrative (never fails, never fabricates)
```

## 🔄 Multi-Provider LLM Gateway (kept from the original starter, extended)

- **Providers**: Gemini (up to 4 rotated keys + a quality-fallback model), Groq (primary +
  fallback model), OpenAI, Mistral, Cerebras — all via LangChain's `ChatOpenAI`, since every
  one of them exposes an OpenAI-compatible endpoint.
- **Automatic failover**: a 429/5xx/timeout puts a deployment on cooldown and moves to the
  next; a 404 (decommissioned model) or invalid API key disables that deployment outright.
  Every switch emits a `ProviderStatusEvent` the caller can surface ("Switched to Groq...").
- **Two call shapes, one fallback loop**: `generate()` for free-form text, and
  `generate_structured()` for schema-validated output (what the summary feature uses) — both
  share the same deployment-selection/cooldown/error-classification loop
  (`LLMGateway._run_with_fallback`), so adding a third call shape later doesn't mean
  reimplementing failover again.
- **Structured output via LangChain tool-calling**: `generate_structured()` uses
  `with_structured_output(schema, method="function_calling", include_raw=True)` — tool-calling
  rather than OpenAI's native strict JSON-schema mode, because the latter isn't portable across
  the other OpenAI-compatible endpoints this gateway fans out to.

## 📁 Project Structure

```
src/
├── app/
│   ├── main.py                       # FastAPI app: CORS, route mounting, root index
│   │
│   ├── api/
│   │   ├── router.py                 # Mounts /api/summary and /health
│   │   └── routes/
│   │       ├── summary.py            # POST /api/summary/generate
│   │       └── health.py             # GET /health — gateway deployment status
│   │
│   ├── core/
│   │   ├── config.py                 # Pydantic settings (providers, gateway, summary knobs)
│   │   └── logging.py                # Console logging (UTF-8 safe on Windows)
│   │
│   ├── gateway/                      # Multi-provider LLM Gateway & failover
│   │   ├── gateway.py                # generate() + generate_structured(), shared fallback loop
│   │   ├── deployment.py             # Per-deployment credentials/cooldown state
│   │   ├── error_classifier.py       # Retryable vs non-retryable error classification
│   │   └── status.py                 # ProviderStatusEvent
│   │
│   ├── schemas/
│   │   ├── common.py                 # UsageInfo, ProviderStatusEventSchema
│   │   └── summary.py                # StructuredSnapshot, SummaryNarrative, response models
│   │
│   └── services/
│       └── summary_service.py        # Snapshot → prompt → generate → validate → response
│
├── tests/
│   ├── test_gateway.py               # Fallback, cooldown, disablement, structured output
│   ├── test_summary_service.py       # No-activity short-circuit, validation, fallback template
│   └── test_api.py                   # /, /health, /api/summary/generate
│
├── run.py                            # Server entry point
├── pyproject.toml                    # Dependencies & tool config
└── .env.example                      # Environment variables template
```

---

## ⚡ Quickstart

### 1. Prerequisites
Ensure you have [uv](https://docs.astral.sh/uv/) installed:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Setup Environment
```bash
cp .env.example .env
uv sync
```

### 3. Configure at least one LLM provider
Edit `.env` — every provider is optional, but the service needs at least one configured or
`/api/summary/generate` will fall back to the deterministic template for every request
(harmless, but you won't be exercising the AI path). See `.env.example` for the full list
(Gemini, Groq, OpenAI, Mistral, Cerebras).

### 4. Run the server
```bash
uv run python run.py
# or: uv run uvicorn src.app.main:app --reload
```

- 📖 Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- ✅ Health: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 API

### `POST /api/summary/generate`

```bash
curl -X POST http://localhost:8000/api/summary/generate \
  -H "Content-Type: application/json" \
  -d '{
    "snapshot": {
      "workspace_id": "ws_123",
      "workspace_name": "Design Team",
      "summary_date": "2026-09-16",
      "timezone": "Asia/Karachi",
      "total_focused_seconds": 7200,
      "members": [
        {
          "user_id": "user_1",
          "display_name": "Amina",
          "focused_seconds": 7200,
          "tasks": [
            {"task_id": "t1", "name": "Redesign homepage", "status": "completed", "focused_seconds": 5400},
            {"task_id": "t2", "name": "Review PRs", "status": "in_progress", "focused_seconds": 1800}
          ]
        }
      ]
    }
  }'
```

Response: the input `snapshot` echoed back, a `narrative` (`overall_summary`, per-member
`member_notes`, `highlights`), and `meta` (`provider`, `model`, `usage`, `status_events`,
whether the deterministic `used_fallback_template` was needed, and any
`validation_warnings`). This service is **stateless** — it never touches a database.
Idempotency, the `(workspace_id, summary_date)` uniqueness constraint, regeneration
versioning, and RLS all live in the OnTask Next.js app that calls this endpoint (see
`project_document/ontask-evolution-plan.md`, Phase 10, in the main OnTask repo).

### `GET /health`
Reports overall status and every configured provider deployment's availability/cooldown state.

---

## 🧪 Testing

```bash
uv run pytest            # all tests, fully mocked — 0 real API tokens consumed
uv run pytest -v tests/test_summary_service.py
uv run pytest --cov=src.app tests/
```

---

## ☁️ Deploying to Vercel

`api/index.py` re-exports the FastAPI `app` from `src/app/main.py`, and
`vercel.json` points Vercel's Python runtime (`@vercel/python`) at it — so
`vercel --prod` (or a Vercel Git integration) deploys this service as-is,
no separate build step needed.

Steps:
1. Import this repo as its own Vercel project (separate from the Next.js
   app's project — they deploy independently).
2. Set every variable from `.env.example` in that Vercel project's
   Environment Variables settings — `.env` itself is never read in
   production, only local dev.
3. Set `ALLOWED_ORIGINS` to your deployed Next.js app's real origin(s) (e.g.
   `https://ontask-by-abrar.vercel.app`), not just `localhost`.
4. Point the Next.js app's own `ONTASK_LLM_SERVICE_URL` (in *its* Vercel
   project's env vars) at this service's deployed URL.

Two things worth knowing about running this specific service on Vercel's
serverless runtime, rather than as a long-lived process:

- **Provider cooldown tracking resets on every cold start.** `LLMGateway`
  tracks rate-limit cooldowns and disabled deployments as in-memory state on
  its singleton instance (see `gateway.py`) — that state doesn't survive
  between separate serverless invocations the way it would on a persistent
  server. Failover still works correctly within a single warm invocation;
  it just won't "remember" a cooldown across cold starts. Not a correctness
  issue, just reduced effectiveness of that specific optimization.
- **Function timeouts.** Each fallback attempt in `_run_with_fallback` has
  its own 30s per-provider timeout, and `GATEWAY_MAX_ATTEMPTS` defaults to
  10 — a worst-case chain of failures could exceed Vercel's default function
  timeout (10s on Hobby, higher on Pro/Enterprise). Consider lowering
  `GATEWAY_MAX_ATTEMPTS` (e.g. 3-4) via that Vercel project's env vars if
  you're on a plan with a short timeout ceiling.

---

## 🔐 Security Considerations

- API keys stored in `.env` (never committed).
- No cross-workspace data ever enters this service — the caller is responsible for only
  sending a snapshot for a workspace the requesting user belongs to (Phase 10.7).
- Type validation with Pydantic on every request and every model response.
- CORS restricted to `ALLOWED_ORIGINS`.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Built By

**[Abrar Ahmed](https://www.abrarahmed.pro)** — AI Engineer & Full-Stack Developer

<a href="https://www.abrarahmed.pro" target="_blank">
  <img src="https://www.abrarahmed.pro/assets/devAbby-fulllogo-C9-MX7QK.png" alt="devAbby logo" height="50" />
</a>
