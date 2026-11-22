# Catalog Build — Decisions

**Go: start batch (a) on a new branch.** Resolutions below. (b) next, (c) with the backend,
item 7 held pending model slugs/prices.

## Batch (a) — config, registry, catalog fn, shim, non-blocked UI

1. **Seedream 5.0 Flash → `basic`.** Your table pairs it with a Basic run (row 4), so Basic
   users get it. Cumulative tiers then give it to everyone above.
2. **Muse Image → don't offer while `available: false`.** Can't serve what no endpoint
   serves. The `catalog(plan)` function returns only `available: true` models, so Muse
   reappears on its own if an endpoint comes back. Drop it from the shown Basic list for now.
3. **Nano Banana 2.1 → drop from the offered Ultra set for now.** Your 20-combination table
   omitted it and it has no measured price. Keep it in the registry, unoffered; add it
   deliberately with a confirmed price if you want it.
4. **Aspect ratios → add 4:5 and 3:2; keep 1:1, 3:4, 16:9, 9:16.** Adding 4:5 fixes the real
   bug (the Brief offers 4:5, the presets don't → silent fallback to 1:1). Rule: **every
   ratio the Brief can emit must exist in the presets.** Keep 4:3 or drop it — your call, it
   doesn't affect the bug.
5. **4K → Ultra only; Max up to 2K.** Keep the code's `setting_policy`. 4K is a cost and
   upsell lever, and reserving it for Ultra is the stronger product line. I'll update the
   catalog spec to match the code (the spec was looser).
6. **Image count → per-plan cap: Basic 2, paid 4.** Replace the flat cap of 4.
10. **Catalog → build `catalog(plan)` from the registry** (filtered by availability + plan),
    exposed via the **dev shim only**. The real HTTP endpoint is the backend's job.

Frontend changes that don't depend on open items (do in (a)): `operation` buttons with
"coming soon", per-stub escalate message, drop the "too complex" UI, the image-count cap, the
mode toggle, images shown one at a time.

## Batch (b) — model pick (item 8)
Add `model` and `stack` request fields; the gateway starts the chain at the chosen model,
**validated against the plan** (reject a disallowed pick before any spend). The picker and
stack-picker UI depend on this.

## Batch (c) — manual mode (item 9), with the backend
Needs a schema change (`MissingItem` gains 3 options + a recommended flag), new `question`
and `error` SSE events, and a **backend-side pause + 5-minute timer**. Coordinate with the
backend teammate. Interim: keep today's NEEDS_INPUT + frontend-resubmit; the frontend shows
the question without the timed pause.

## Item 7 — reasoning stacks: HOLD, with direction
- **Gate fixed at `gpt-oss-20b` across all stacks: correct, keep it.**
- **R1 (free/Basic) must stay free Groq, one model per agent.** Two fixes before it's usable:
  - Don't reuse the Gate's `gpt-oss-20b` for the Brief — keep the Brief on a different Groq
    model (`gpt-oss-120b`), or you halve the per-model free quota.
  - Run the Critic (`llama-4-scout`) **on Groq, not Cloudflare** — Cloudflare spends the
    shared image-neuron pool. If `llama-4-scout` isn't free on Groq, use a Groq vision model
    (`qwen3.8-27b`) for the free stack. Confirm its Groq availability.
  - Net: the free stack must cost ~$0 and touch no image neurons.
- **R2–R4 are paid.** `ALLOW_PAID` governs them — keep it off in dev. Build them only after
  you give the **exact slugs and measured prices** for GPT-6 Luna, MiniMax M3, Kimi K2.6 and
  Step 3.7 Flash; none are in `chains.yaml` yet.
- **CLAUDE.md wins on conflict.** Change the one-model-per-agent rule only by editing CLAUDE.md
  deliberately — and I'd keep it for the free stack regardless.

## Order
(a) now → (b) → (c) with the backend. Hold 7 until the slugs/prices and the CLAUDE.md call
are settled.
