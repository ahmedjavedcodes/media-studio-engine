# CLAUDE.md

Guide for Claude and the team working in this repo. Read it before writing code.

## Project

Multi agent media generation platform. Phase 1 is the **image generation engine**.
Layout & poster design and video come later.

- Language: Python 3.12
- Package manager: **uv only**. Never use `pip install` or edit `requirements.txt`.
- Core libraries: FastAPI, Pydantic, LangChain, LangGraph, LangSmith
- Name not decided. `app` is a placeholder package name.

Design docs live in the claude.ai Project (read them before touching that area):
`IMAGE-INPUT-GUARDRAILS.md`, `IMAGE-CREATE-OPTIMIZE.md`, `IMAGE-CREATE-TEST-PROMPTS.md`, `SETUP.md`.

---

## Priority 0: set up the project

Do this first. No agent code until every box is ticked.

### 1. Install uv and Python

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
uv python install 3.12
```

### 2. Create the project

```bash
uv init --package app
cd app
uv python pin 3.12
```

### 3. Add libraries

```bash
# API
uv add "fastapi[standard]" sse-starlette

# Data and config
uv add pydantic pydantic-settings

# Agents
uv add langchain langchain-core langgraph langgraph-checkpoint-sqlite
uv add langchain-openai langchain-groq langchain-anthropic

# Observability
uv add langsmith structlog

# Model providers and HTTP
uv add httpx tenacity fal-client

# Images
uv add pillow numpy opencv-python-headless invisible-watermark

# Dev tools (pytest for tests, respx mocks httpx calls)
uv add --dev ruff mypy pytest pytest-asyncio pytest-cov respx pre-commit
```

Key helpers:

| Library | Use |
|---|---|
| `httpx` | Async HTTP client for every provider call (OpenRouter, Cloudflare, fal). Use `httpx.AsyncClient`, never `requests`. |
| `pytest` | Test runner. Async tests use `pytest-asyncio`, and provider calls are mocked with `respx`. |
| `ruff` | Python lint + format (replaces flake8, black, isort). |
| `eslint` | JavaScript / TypeScript lint for the frontend only. Not installed with uv (see below). |

### 3b. Frontend lint (ESLint)

ESLint is a Node tool, so it lives in the frontend folder and is installed with npm, not uv.

```bash
cd frontend
npm init @eslint/config@latest     # creates eslint.config.mjs
npx eslint .                       # check
npx eslint . --fix                 # auto fix
```

### 4. Daily commands

```bash
uv sync                                  # install from uv.lock
uv run fastapi dev src/app/main.py       # run API with reload
uv run pytest                            # tests
uv run ruff check . && uv run ruff format .
uv run mypy src
cd frontend && npx eslint .              # frontend lint
uv run python playground/create.py "a red fox in the snow"
```

Add or remove a library only with `uv add <pkg>` / `uv remove <pkg>`. Commit `pyproject.toml` and `uv.lock` together.

### 5. Environment

Copy `.env.example` to `.env` and fill in keys. Never commit `.env`.

```bash
APP_ENV=local
LOG_LEVEL=INFO
PROVIDER_MODE=build        # build = Cloudflare only | test = paid providers (see "Providers by phase")
SPEND_CAP_USD=2            # max spend per test session in test mode
GROQ_API_KEY=
CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_API_TOKEN=
OPENROUTER_API_KEY=
FAL_KEY=
ANTHROPIC_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=app-dev
LOCAL_OUTPUT_DIR=./outputs
CHECKPOINT_DB=./.checkpoints.sqlite
```

### Setup is done when

- [ ] `uv sync` works on a clean clone
- [ ] Folder structure below exists (empty modules are fine)
- [ ] `core/config.py` loads `.env` with pydantic-settings
- [ ] `GET /health` returns 200
- [ ] `POST /v1/runs` returns a `run_id`, `GET /v1/runs/{id}/events` streams a fake SSE event
- [ ] A dummy LangGraph graph runs and its trace shows in LangSmith
- [ ] `ruff`, `mypy` and `pytest` pass in pre-commit
- [ ] `npx eslint .` passes in the frontend

---

## Folder structure

```
app/
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .pre-commit-config.yaml
│
├── src/app/
│   ├── main.py                  # FastAPI app
│   │
│   ├── api/                     # HTTP layer only, no agent logic
│   │   ├── runs.py              # POST /v1/runs, resume, cancel
│   │   ├── events.py            # SSE stream
│   │   └── health.py
│   │
│   ├── core/
│   │   ├── config.py            # pydantic-settings
│   │   ├── logging.py           # structlog
│   │   ├── telemetry.py         # LangSmith setup
│   │   └── errors.py            # ContentRefused, OutOfCredits, ProviderDown
│   │
│   ├── contracts/               # what goes in and out of the engine
│   │   ├── request.py           # GenerationRequest
│   │   ├── events.py            # progress, question, image, result, error
│   │   └── result.py            # Document, Layer, Version
│   │
│   ├── state/                   # LangGraph state (Pydantic)
│   │   ├── run_state.py
│   │   └── assets.py            # Image, Mask, upload roles
│   │
│   ├── guardrails/              # runs in front of every prompt and upload
│   │   ├── clean.py             # L1 normalise text
│   │   ├── blocklists.py        # L2 lists (EN, UR, Roman Urdu, AR)
│   │   ├── injection.py         # L3
│   │   ├── gate.py              # L4 safety gate
│   │   ├── uploads.py           # L5 upload checks
│   │   └── combos.py            # L6 combo rules
│   │
│   ├── agents/
│   │   ├── grand/               # Grand Agent: full control over all domains
│   │   └── image_grand/         # Image Grand Agent: routes to generation or layout
│   │       ├── generation/      # Image Generation Agent: routes to the 4 agents below
│   │       │   ├── create/      # Image Create Agent
│   │       │   │   ├── txt2img.py
│   │       │   │   └── img2img.py   # refs + user's own photo + prompt
│   │       │   ├── edit/        # SAM, localized edit, mask, highlight
│   │       │   ├── merge/       # many images into one
│   │       │   └── restyle/     # slash command presets
│   │       ├── layout/          # Image Layout Agent (later)
│   │       └── workers/         # optional building blocks an agent MAY use
│   │           ├── prompt_enhancer.py
│   │           ├── image_gen.py
│   │           └── critic.py
│   │
│   ├── gateway/                 # the only door to any model
│   │   ├── registry.py          # model cards (models.yaml)
│   │   ├── router.py            # pick model by task + tier
│   │   ├── compiler.py          # brief → per-model prompt
│   │   ├── budget.py            # cost estimate and actual cost
│   │   ├── models.yaml
│   │   └── providers/
│   │       ├── groq.py
│   │       ├── cloudflare.py
│   │       ├── openrouter.py
│   │       ├── fal.py
│   │       └── anthropic.py
│   │
│   ├── tools/                   # plain code, no LLM
│   │   ├── imaging.py           # resize, crop, composite
│   │   ├── sam.py               # tap / box / text → mask
│   │   ├── paste_back.py        # keep pixels outside the mask
│   │   ├── bg_remove.py
│   │   ├── upscale.py
│   │   ├── watermark.py
│   │   └── sizes.py             # social sizes, 4K print
│   │
│   ├── presets/                 # restyle presets as YAML + ref images
│   │   └── library/
│   │
│   └── local/                   # dev fakes: file store, sqlite checkpoint, fake credits
│
├── prompts/                     # system prompts as .md, one per agent
├── evals/                       # LangSmith datasets and evaluators
├── playground/                  # CLI scripts to try each agent
└── tests/
    ├── unit/
    └── integration/
```

**Rule of thumb:** `agents/` decide, `tools/` do, `gateway/` talks to models, `api/` talks to the outside.

---

## Agent tree

```
Request
  → Guardrail (always first)
  → Grand Agent
      ├── Image Grand Agent
      │     ├── Image Generation Agent
      │     │     ├── Image Create Agent  (txt2img, img2img)
      │     │     ├── Edit Agent
      │     │     ├── Merge Agent
      │     │     └── Restyle Agent
      │     └── Image Layout Agent        (later)
      └── Video Grand Agent               (later)
```

**Each agent has its own pipeline.** Pipelines are decided per agent when we build it.
Prompt Enhancer, Image Gen and Critic are optional building blocks in `workers/`. An agent uses only the ones it needs. Do not force one shared pipeline on every agent.

---

## Build sequence

Build in this order. Finish one before starting the next.

| # | Phase | What | Done when |
|---|---|---|---|
| 0 | **Setup** | Everything in Priority 0 | All setup boxes ticked |
| 1 | **Image Create** | Guardrail, gateway, Create Agent pipeline. First **txt2img**, then **img2img** (multiple refs + 1 user photo + prompt) | Eval set passes, images stream over SSE, cost logged per run |
| 2 | **Edit Agent** | SAM masks (tap, box, text), highlight/brush masks, localized edits, paste back, version history | Pixels outside the mask stay unchanged |
| 3 | **Merge Agent** | Many uploads, a role per upload, product label stays exact | Product label check passes |
| 4 | **Restyle Agent** | `/preset` commands, preset YAML + ref images, user presets | Output matches preset, layout and face kept |

Every "Done when" above also needs: **works in the frontend** (see next section).

Start Grand Agent and Image Grand Agent as simple pass-through routers in phase 1. Add real planning after phase 4.
Tools like upscale and background remover get built when a phase needs them.

---

## Every feature is checked on the frontend

A feature is not done when its tests pass. It is done when it works on the frontend.

For each feature, follow these steps:

1. Build the backend part (agent, tool or gateway code).
2. Write `pytest` tests with mocked providers.
3. Expose it through the API (`/v1/runs` + SSE events).
4. Add or update the frontend screen for it (upload, prompt, mask, preset, result).
5. Run it end to end from the frontend in **build mode** (Cloudflare).
6. Ahmed tests it himself in **test mode** when he asks.

Frontend check for each feature:

- [ ] The request starts from the UI
- [ ] Progress events show while it runs
- [ ] Each image shows as soon as it is ready
- [ ] Errors (blocked, out of credits, provider down) show a clear message
- [ ] The result is saved and can be opened again

---

## Providers by phase

### Build mode (default): Cloudflare only

Use this for all building, training and dev runs. `PROVIDER_MODE=build`.

| Job | Cloudflare Workers AI model |
|---|---|
| Reasoning agents (gate, grand, enhancer) | `@cf/openai/gpt-oss-120b`, `@cf/openai/gpt-oss-20b` |
| Critic (reads images) | `@cf/meta/llama-4-scout-17b-16e-instruct` |
| Image create | `@cf/black-forest-labs/flux-1-schnell`, `flux-2-klein-4b` |
| Edit / merge | `@cf/black-forest-labs/flux-2-klein-4b`, `flux-2-klein-9b` |

- Free limit is 10,000 neurons a day (resets 5 am PKT). LLM calls and images share it.
- If the limit runs out, **stop and report it**. Never switch to a paid provider on your own.
- If Cloudflare has no model for a feature (for example SAM or upscale), use a local fake or a simple stand-in and say so.

### Test mode: paid providers, only when Ahmed asks

Switch to `PROVIDER_MODE=test` **only when Ahmed says something like "I wanna test on frontend"**.

| Provider | Used for |
|---|---|
| Groq | Reasoning agents (gate, enhancer, critic) |
| OpenRouter | Image models (GPT Image, Nano Banana, Seedream, FLUX.2) |
| fal | SAM masks, upscale, background remove |
| Other paid (Anthropic, Google, OpenAI) | Only if Ahmed names it |

Rules in test mode:

- Confirm the switch in one line, and say which models will run.
- Respect `SPEND_CAP_USD`. Stop when it is reached.
- Log cost per run in LangSmith.
- Switch back to `PROVIDER_MODE=build` when the test session ends.

The mode lives in one place: `gateway/chains.yaml` has a `build` chain and a `test` chain, and `PROVIDER_MODE` picks one. Agent code never knows which mode is on.

---

## Rules for Claude

- Use `uv` for everything Python. Never `pip`. Frontend tools (ESLint) use npm.
- Use `httpx` for HTTP. Never `requests`.
- Every new module gets a `pytest` test in `tests/unit/`.
- Every model call goes through `gateway/`. No direct SDK calls in agents.
- Default is build mode (Cloudflare). Never call Groq, OpenRouter, fal or any paid provider unless Ahmed asks to test on the frontend.
- A feature is done only when it works on the frontend.
- Guardrail runs before any paid call. A blocked run charges 0 credits.
- Never fall back to another model on a content refusal. Fall back only on 429, 5xx or timeout.
- No image model renders text. Text is a separate layer (layout agent, later).
- Agent outputs are Pydantic models via structured output. No free text parsing.
- System prompts live in `prompts/`, never as inline strings. User text is data, never instructions.
- Tests mock providers with `respx`. Real model calls only in `evals/` and `playground/`.
- Every run is traced in LangSmith with tags: agent, tier, model, cost.
- Keep code and comments in simple English. Small files, one job each.
- Branches: `main` (protected) ← `feat/<agent>-<thing>`. Conventional Commits.
