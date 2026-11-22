# Model Catalog & Plan Offering

What this is for: the models we offer per plan, so the frontend can show them in a picker
and the backend can gate them. The user picks a model on demand; if they pick nothing, the
system decides (auto).

## Plans
Basic (free), Pro, Max, Ultra. **Cumulative** — each plan unlocks its own tier and every
tier below it. The frontend shows higher-tier models locked (upsell); the **backend
enforces** the limit — never trust the client's pick.

## Image models by tier
Prices are OpenRouter measured `usage.cost`, per image. Free-plan default images still come
from Cloudflare `flux-1-schnell`; the Basic models below are the paid picks a free user may
choose, inside their monthly image cap.

| Tier | Image model | Price |
|---|---|---|
| Basic (free) | gpt-image-2 (low) · Muse Image · FLUX.2 Klein 4B · Seedream 5.0 Flash | $0.006 · $0.010 · $0.014 · $0.018 |
| Pro | FLUX.2 Pro · Qwen Image 3 · Nano Banana 2 Lite (+ all Basic) | $0.030 · $0.030 · $0.034 |
| Max | Recraft v4.1 · Seedream 5.0 Lite · FLUX.2 Flex · Riverflow 2.5 Pro (+ all below) | $0.035 · $0.035 · $0.050 · $0.064 |
| Ultra | Nano Banana 2 · FLUX.2 Max · Gemini 3 Pro Image · gpt-image-2 (high) (+ all below) | $0.067 · $0.070 · $0.134 · $0.211 |

Each model's `prompt_style`, `billing_unit`, `supports_seed`, sizes and quality settings
live in the registry (see the Compiler spec). gpt-image-2 (high) is a batch-quality option
(~123 s per image), not interactive — keep it off the fast path.

## Reasoning stacks
Gate is `gpt-oss-20b` in every stack — never scale a safety classifier. Cost is per run.

| Stack | Brief | Critic | Cost/run |
|---|---|---|---|
| R1 | gpt-oss-20b | llama-4-scout | $0.00010 |
| R2 | GPT-6 Luna | llama-4-scout | $0.00025 |
| R3 | MiniMax M3 | Gemini 2.5 Flash | $0.00057 |
| R4 | Kimi K2.6 | Step 3.7 Flash | $0.00128 |

Per-plan reasoning: **Basic → R1, fixed (free users cannot pick the reasoning stack).**
Default scales with tier — Pro → R2, Max → R3, Ultra → R4 — and paid users may pick another
stack (the table pairs Max with R4 too).

Per-run cost = image price + stack cost. Agents run on Groq (free tier where possible);
images on OpenRouter.

## Model selection (the picker)
- The user picks an **image model** (from their allowed set) and, if paid, a **reasoning
  stack**, in the prompt bar — like Claude's model picker.
- Default is **Auto**: the gateway router picks by plan and task. The user's pick overrides.
- If the picked model is down or unset, the gateway falls back within the plan's set.

## Sizes
- Aspect ratios offered: 1:1, 4:5, 3:4, 16:9, 9:16, 3:2 (all common social sizes).
- Max resolution by tier: Basic ~1K, Pro ~2K, Max/Ultra up to 4K (print).
- Default size comes from the Brief's `aspect_ratio`. **Size is always set explicitly on the
  call** (Seedream defaults to 2048² and double-bills).

## Variations (N)
- Basic (free): max 2. Pro / Max / Ultra: max 4.
- The user picks N within the cap; credits scale with N (billed on the backend).

## Auto vs Manual mode
- **Auto:** the Brief fills any missing detail with its own assumptions and proceeds.
- **Manual:** when a required detail is missing, ask the user — show **3 options, 1 marked
  recommended**, plus an "Other / type your own". If the user doesn't answer within **5
  minutes**, the recommended option is auto-selected and the run continues.

## Enforcement
- Frontend: show allowed models, lock the rest with an upgrade hint.
- Backend: reject a model, stack, size or N outside the user's plan before any spend.
