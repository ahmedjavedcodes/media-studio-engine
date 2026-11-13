# CLAUDE.md

## Scope

This repo is the **AI side** of our agentic media studio: model gateway, agents, image services and harness. A teammate builds the traditional backend (API, auth, database, payments, job queue) in a separate repo. **Do not build any of that here.** See SETUP.md for the layout and the contract (section 8).

## Secrets

All API keys are in `.env`. Never print, log or commit them. `.env` must stay in `.gitignore`. Do not read `.env` contents into output; `.env.example` documents the variable names.

## Model provider strategy

Rule: Cloudflare Workers AI is the first choice for EVERY task (agent LLMs and images) during the build phase. When its daily limit runs out, fall back to Groq (free, LLMs only), then to paid providers. Agent code must never call a provider directly. Every model call goes through `gateway/`, which picks the provider from a fallback chain.

### Providers and .env keys

- **Cloudflare Workers AI** (free: 10,000 neurons/day, shared by LLMs AND images; resets 00:00 UTC)
  - Keys: `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`
  - LLMs (OpenAI-compatible): `https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1/chat/completions`
  - Images: `POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/<model>` (returns base64)
- **Groq** (free tier, limits per model: 30 RPM, 1K req/day, 8K TPM, 200K tokens/day)
  - Key: `GROQ_API_KEY`. OpenAI-compatible base URL: `https://api.groq.com/openai/v1`
  - Reasoning models: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-safeguard-20b` (safety checks)
  - `reasoning_effort`: low/medium/high. `reasoning_format` is not supported on GPT-OSS; use `include_reasoning` there. Never use `reasoning_format: "raw"` with tools or JSON mode.
- **Paid fallbacks**
  - OpenRouter (`OPENROUTER_API_KEY`): images via `POST https://openrouter.ai/api/v1/images` (`input_references` for edit/merge); LLMs via `/api/v1/chat/completions`
  - Anthropic Claude (`ANTHROPIC_API_KEY`): designer agent and the hardest planning
  - fal (`FAL_KEY`): SAM masks, background removal, upscalers
  - Google AI / OpenAI: last-resort direct fallback
- **TypeSafe Jev** (`TYPESAFE_API_KEY`, early access): typed decisions (Choice / Score / Noul / Box) behind a `Decider` interface, with an LLM + Pydantic structured-output fallback when the key is empty.

### Fallback chains (`gateway/chains.yaml`, config not code)

```yaml
reasoning:        # planner, brief, router, critic
  - cloudflare/@cf/openai/gpt-oss-120b
  - groq/openai/gpt-oss-120b
  - openrouter/<paid>
fast_llm:         # prompt agent, small tasks
  - cloudflare/@cf/openai/gpt-oss-20b
  - groq/openai/gpt-oss-20b
  - groq/qwen/qwen3.8-27b
  - openrouter/<paid>
designer:
  - cloudflare/@cf/openai/gpt-oss-120b
  - groq/openai/gpt-oss-120b
  - anthropic/claude
image_create:
  - cloudflare/@cf/black-forest-labs/flux-1-schnell
  - cloudflare/@cf/black-forest-labs/flux-2-klein-4b
  - openrouter/<paid>
image_edit:
  - cloudflare/@cf/black-forest-labs/flux-2-klein-4b
  - openrouter/<paid edit model>
image_merge:
  - cloudflare/@cf/black-forest-labs/flux-2-klein-9b
  - openrouter/<paid multi-reference model>
```

Leave the `<paid>` slots as config placeholders. Plan tier may override a chain later.

### When to switch provider (`gateway/fallback.py`)

- Cloudflare HTTP 429 + code 3036 -> daily neurons used up: mark Cloudflare exhausted until 00:00 UTC, skip to next, don't retry
- Cloudflare HTTP 429 + code 3040 -> temporary capacity: retry once with backoff, then next
- Cloudflare code 5035 -> model needs the paid plan: drop that model from free chains and log a warning
- Groq HTTP 429 -> honour `retry-after` if it's a few seconds, otherwise next
- HTTP 5xx / timeout -> retry once, then next
- HTTP 400 -> do NOT fall back (it's our bug): raise a clear error
- Log every call: task, provider, model, latency, cost/neurons, and whether it was a fallback (structlog + LangSmith)

## Rules

- Python 3.12, uv, Pydantic v2, LangGraph, async httpx, tenacity for retries
- Image models never render text; text is always a separate designer layer
- Tests mock all providers with respx. Real calls only in `evals/` and `playground/`.
- Ask before adding new dependencies or providers.
