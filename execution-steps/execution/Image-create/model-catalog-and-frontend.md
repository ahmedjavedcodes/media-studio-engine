# Model Catalog & Frontend Update

Part 1 is the catalog (what we offer per plan). Part 2 is what the existing frontend needs
to change to match the new architecture.

---

# Part 1 — Model Catalog

The models we offer per plan, so the picker can show them and the backend can gate them. The
user picks a model on demand; if they pick nothing, the system decides (auto).

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

gpt-image-2 (high) is a batch-quality option (~123 s per image), not interactive — keep it
off the fast path.

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
stack (the table pairs Max with R4 too). Per-run cost = image price + stack cost.

## Sizes
- Aspect ratios: 1:1, 4:5, 3:4, 16:9, 9:16, 3:2 (common social sizes).
- Max resolution by tier: Basic ~1K, Pro ~2K, Max/Ultra up to 4K (print).
- Default size comes from the Brief's `aspect_ratio`. Size is always set explicitly on the
  call (Seedream defaults to 2048² and double-bills).

## Variations (N)
Basic (free): max 2. Pro / Max / Ultra: max 4. The user picks N within the cap; credits
scale with N (billed on the backend).

## Auto vs Manual mode
- **Auto:** the Brief fills any missing detail with its own assumptions and proceeds.
- **Manual:** when a required detail is missing, ask the user — 3 options, 1 recommended,
  plus an "Other / type your own". No answer in 5 minutes → the recommended is auto-selected.

---

# Part 2 — Frontend Update

The existing frontend is on the old architecture. These are the changes to bring it in line.
Don't test the new pipeline through the old UI — update it first, or test the backend directly.

## Model picker
- **Fetch the catalog from the backend** (registry-backed); don't hardcode the list. Show
  the plan's allowed models; lock higher tiers with an upgrade hint. Default = Auto.
- Add a **reasoning-stack picker for paid users** (R1–R4). Free is locked to R1 — hide or
  disable it.

## New request fields
- `operation` — set by a button or slash command. `create` is live; edit / merge / restyle
  show "coming soon". Default `create`.
- **Size / aspect ratio** picker, from the plan's sizes.
- **N variations** — within the plan cap (2 free, 4 paid).
- **Mode** — auto or manual.

## Behaviour changes to handle
- **"Complex" prompts no longer escalate.** Drop any "too complex" UI tied to the old route
  — a multi-step prompt now just returns one image.
- **Stub operations escalate.** edit / merge / restyle return ESCALATE — show a clear
  per-operation "not available yet" message, not the old generic escalate text.

## Manual-mode flow
On a `question` event, render 3 options + 1 recommended + an "Other / type your own". If the
user doesn't answer in 5 minutes the backend auto-picks the recommended — reflect that in the
UI (don't hang waiting).

## SSE / result
Consume `progress` / `question` / `result` / `error`. Images **stream in as they finish** —
show them one at a time, not all at the end. The result is a document with layers.

## Gating
UI locks are UX only; the backend enforces the plan. Expect a rejection if a user forces a
disallowed model, size or N — handle the error gracefully.
