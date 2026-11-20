# IMAGE-CREATE-OPTIMIZE-2.md: the work left after session 1

What is left of [IMAGE-CREATE-OPTIMIZE.md](IMAGE-CREATE-OPTIMIZE.md) (v2.2), written for whoever picks it up next. Section numbers like §5.4 refer to that spec. Progress so far is in [IMAGE-CREATE-OPTIMIZE-CHECKLIST.md](IMAGE-CREATE-OPTIMIZE-CHECKLIST.md).

---

## 0. Session 2 status (uncommitted on `image-create`; 654 passed, 3 skipped; ruff, mypy, tsc clean)

| item | state |
|---|---|
| M1 Groq checks | Done. `json_schema` works on all four Groq models with the real schemas (`ModelCard.json_schema`). Providers without it get the schema appended to the system prompt (`inject_schema`). qwen takes `reasoning_effort` `low` and `none`; Critic uses `none` (0 reasoning tokens). gpt-oss still spends 4-28 at `low`, so "0 reasoning tokens when low" cannot hold for it. |
| M1 budgets | Done: `tests/unit/harness/test_prompt_budgets.py` (Gate 199/200, Brief 452/600, Critic 148/150, safety 185/200). Needs `tiktoken`, which is installed but NOT declared in pyproject (skips if missing): ask before adding it. |
| M2 ranking Critic | Done. One call for all images, reads stored 512 px thumbnails, per-image block, ranks closed in code, `scores` and `rank` on records and documents. Old single-image path removed. E4 (position bias) not run. |
| M3 Brief + Compiler | Done. Compiler runs per chain entry inside the gateway walk (`Gateway._compiled_for`). Skipping the Brief on `prompt_is_detailed` NOT built: it would lose `must_have` and the aspect rule until E6 says it is safe. |
| M4 budget | Done. USD estimate enforced per paid call (`max_cost_usd`), unknown price = cannot afford, `USD_ESTIMATE_ENFORCED` True. Generate step cost now the dearest plan model. Neurons 57.6. |
| M5 Klein seed | Run: 2 calls, $0.028, **different pixels**. No model reproduces from a seed. Registry: `seed_reproduces: false`. Ledger total **$0.0340 of $5.00**. |
| M6 | `uv lock` done. RunState deprecated fields and `image_ids` untouched (need the backend teammate). `execution-steps/` still untracked. |

Not run (need approval): E1, E3-E7, M7 live measurements. E2 approved but its compiled-prompt arm can now run.

---

## 1. Where things stand

**Branch `image-create`**, pushed. Commits from session 1:

| commit | what |
|---|---|
| `817217b` | Groq first for agents, one Groq model per agent |
| `23cd975` | OpenRouter image registry (15 models), registry-aware adapter, USD estimate |
| `028d171` | Inspector: provider panel |
| `370e9b3` | Review 1 fixes: plan guard in the gateway, one registry, recorded OpenRouter response |
| `22bfdd0` | Linear create, derived seeds, step 0 code checks, the Gate, review 2 fixes |
| `a984c5b` | Settings screening, Critic in-step retry, settle rule, D4, D5, thumbnails, documents, streaming |

**Done:** build-order items 2, 4, 5 and 6, most of 3, 9 and 10, plus D2, D4 and D5. Create is fully linear.

**Not committed:** the `execution-steps/` folder (the spec, the checklist and this file) has never been tracked. Decide whether it belongs in git.

**How to run things.** These are machine quirks; details are in the user's memory file `dev-env-quirks`.
- Tests: `PYTHONPATH=src .venv/bin/python -m pytest -q`. The baseline is 617 passed, 3 skipped (agents with no prompt).
- Lint: `.venv/bin/ruff check src tests playground scripts` and `.venv/bin/ruff format --check ...`
- Types: `PYTHONPATH=src .venv/bin/python -m mypy src/media_engine`
- Frontend: `cd frontend && npx tsc --noEmit -p . && npx next build`
- `uv` lives at `~/Library/Python/3.14/bin/uv`. The venv has no `pip`.
- `gh` and `git push` need `export PATH="$HOME/.local/bin:$PATH"`.
- If a test run hangs, use `-o faulthandler_timeout=30` to get a stack dump.

---

## 2. Ground rules (they still hold)

- **Spend cap: $5 total across all real paid calls.** Spent so far: **$0.0060**, one gpt-image-2 call. The ledger is `.cache/paid_spend.jsonl`; `playground/record_openrouter_response.py` shows how to check it before calling.
  - Report the running total after every paid call.
  - `ALLOW_PAID` stays off except for an approved eval, then goes off again.
- **Ask before any real call, including free ones** (Cloudflare, Groq). Tests mock every provider with respx.
- **Approved but not yet run:** E2 (about $0.16).
- **Need approval:** E1 and E3–E7, and the paid Klein seed check. E3's `high` arm (about $6.30, about an hour) needs its own explicit approval.
- **Commit and push only when asked.** After each milestone, expect "summary for reviewer". That means a review bundle in the scratchpad: a context header, the design decisions to scrutinise, full source plus every touched test file, and the diff. Scan it for secrets. Apply valid findings, and argue against wrong ones with evidence.
- **Ask before adding dependencies or providers** (CLAUDE.md).
- **`.env`:** never read, print or commit it.

---

## 3. Remaining work, in recommended order

### M1. Two free Groq checks, then the §8 prompt rules

**Why first:** both checks are free and quick, and they unblock the token budgets.

1. **Ask first**, then make two free Groq calls with `playground/smoke_test.py`:
   - Does `openai/gpt-oss-20b` (and `-120b`) on Groq accept `response_format: {type: "json_schema", json_schema: {...}}`, the provider's structured-output mode?
   - Does `qwen/qwen3.8-27b` accept `reasoning_effort: "low"` (or `"none"`)? Check this before sending it. A 400 on the `vision` chain would fail every review, because 400s never fall back.
2. **If structured output works:** send the schema through the provider instead of putting it in the prompt (`{{schema}}` in `agents/base.py` `system_prompt()`).
   - Today the Gate prompt is **452 tokens**: 268 prose and **184 schema**, counted with `o200k_base`. The safety reviewer is 272.
   - Keep the in-prompt schema for providers without structured-output support (the Cloudflare fallback).
3. **Declared token budgets, with a test that fails the build on overrun:** Gate ≤ 200, Brief ≤ 600, Critic ≤ 150 text tokens.
   - Count the rendered system prompt. `tiktoken` (`o200k_base`) is already installed, but its encoding file is downloaded on first use.
4. **`reasoning_effort: low`** for Gate, Brief and Critic, plus a test that reasoning tokens stay at 0 when low.
   - The Critic currently sends no `reasoning_effort`, because of the qwen question above.
5. **At most 2 few-shot examples per prompt**, and no prose description of the schema.

**Done when:** the budget test passes for every agent, and the Gate is at or under 200 tokens.

### M2. The ranking Critic (§5.4, build step 7)

1. **One vision call sees all N images**, as thumbnails, and returns one entry per image:
   - `prompt_adherence`, `craft`, `visual_interest`, `technical`: each 0–10
   - `safety`: allow or block, a **separate binary field, never averaged into the score**
   - `rank`
2. **Put the scale anchors in the prompt** (the spec's §5.4 text), or every image scores 9.
3. **Return critique prose only when a dimension is below threshold.**
4. **Load the rubric by `task_type`** (create now; edit and merge later, §10). Don't hard-code it for create.
5. **Read the stored thumbnails** (`ArtifactStore.get_thumbnail`, 512 and 256 px exist) instead of downscaling to 768 px.
   - Default to 512 until experiment E1 picks a size.
   - Measured: a 768 px JPEG costs about **1,330 prompt tokens** on Groq qwen. Groq allows 8K tokens a minute per model, which is the throughput ceiling.
6. **Keep:**
   - the one in-step retry (`params.retries`), never after a block
   - a block on one image drops that image only (§5.5, already in settle)
   - `image_ready` streaming, which now has to wait for the single ranking call
7. **Fill `documents[].scores`** with the four dimensions; it only has `quality_score` today. Settle should rank by the Critic's `rank`.
8. **Risk:** one call comparing several images can be biased by position. E4 measures this. If it fails, fall back to N separate calls on thumbnails.

**Done when:**
- critic score spread is at least 2 points on a mixed set (needs E5 for real data)
- create makes one Critic call per run, not N

### M3. Brief + Compiler (§4.2, §5.2, build step 8)

1. **The Brief:** the enhancer stops writing a paragraph and returns the §4.2 fields:
   - `subject`, `action`, `setting`, `lighting`, `camera`
   - `detail_anchors`, `must_have`, `aspect_ratio`, `assumptions`

   Already there: temperature 0.2, and `must_have` grounding (`agents/prompt/enhancer.py`).
   To add: `max_tokens: 150`, and the rule that the Brief may not add objects the user didn't ask for.
2. **Skip the Brief when the Gate says `prompt_is_detailed`.** The verdict is recorded on `RunState.gate` but unused. E6 measures whether the skip is safe.
3. **Compiler (`gateway/compiler.py`, plain code, no model)**, from the Brief and the chosen model to that model's prompt:
   - The style template comes from the registry's `prompt_style` (`nouns`, `sentences` or `tags`), plus one for Cloudflare `flux-1-schnell` (the free path).
   - The word limit comes from `prompt_max_words`.
   - Strip the banned vocabulary: pristine, subtle, hints of, fading, muted, soft diffused, gentle, minimal, evoking, atmosphere, balanced, serene, and any trailing mood clause.
   - Require concrete nouns, a light direction, and at least two detail anchors.
   - Aspect ratio never defaults to 1:1: scenes get 16:9, single products or portraits get 1:1 or 4:5.
   - **Re-run the step 0 code checks on the compiled text.**
4. **Open design decision:** the model is picked by the gateway's fallback chain, so the compiler needs the *chosen* entry. Two options:
   - Compile for each chain entry inside the gateway walk, alongside `route_image`. Accurate, but the compiler moves into the gateway layer.
   - Compile once for the first entry. Simple, but wrong after a fallback.

   Decide before building.

**Done when:** every compiled prompt is within its model's word budget, which is an acceptance target.

### M4. Budget fixes: D3, and enforcing the USD estimate

1. **Replace the wrong estimates in `harness.yaml` `step_estimates`:**
   - **Critique: 12,000 tokens** each, sized for a Cloudflare fallback plus a repair. Groq qwen measured about 1,330 prompt tokens. This keeps free n=2 fragile: 27,000 tokens minus the enhancer's real spend must stay at or above 24,000.
   - **Generate: a flat `cost_usd: 0.04`** under-estimates the ultra models: $0.067, $0.134, and $0.211 for gpt-image-2 `high`. That flat figure is what the supervisor's dollar-cap check uses today.
   - **Neurons:** use the verified Cloudflare rates in §2 of the spec. `flux-1-schnell` is 57.6 neurons per 1024² image at 4 steps.
2. **Wire `gateway/budget.estimate_image_usd` into the gateway's pre-call check**, against the plan's `cost_cap_usd`. The `n`-routing and plan guard already run there.
   - Then flip `USD_ESTIMATE_ENFORCED` and update `tests/unit/gateway/test_budget_not_enforcing.py` in the same change.
   - Check the estimate against the recorded real `usage.cost`: gpt-image-2 `low` estimates $0.006 against $0.00596 billed. The fixture is `tests/fixtures/openrouter/gpt-image-2.images.json`.
3. **Treat a `None` estimate as "cannot afford"**, for an unknown price.

### M5. Calls that need approval or money

| item | cost | needs |
|---|---|---|
| Seed reproducibility on OpenRouter FLUX.2 Klein (same seed twice, compare pixels) | ~$0.028 | approval. It is the **only** candidate for reproducible evals, since no free model reproduces (§6 below) |
| E2: prompt vs model quality | ~$0.16 | approved; its compiled-prompt arm needs M3 |
| E1: Critic thumbnail size (512 / 384 / 256 against hand scores) | Groq, free | approval, plus 50 hand-scored images |
| E3: gpt-image-2 low/medium/high vs 3 models | ~$6–8 | approval; the `high` arm separately |
| E4: ranking Critic, position bias | Groq, free | approval; needs M2 |
| E5: Critic calibration, per-tier thresholds | Groq, free | approval; hand-scored set |
| E6: `prompt_is_detailed` precision | Groq, free | approval |
| E7: `n_per_call` vs parallel requests | ~$0.05 | approval |

Then set the model for each tier: the paid `image_create`, `image_edit` and `image_merge` slots in `chains.yaml` are placeholders.

### M6. Decisions and clean-up

- **RunState fields:** `retries_used`, `replans_used`, `critic_reasks_used`, `AttemptRecord.lever` and `.feedback` are deprecated (always 0 or None) but still in the stored-state contract. Remove them with the backend teammate's agreement.
- **`image_ids` vs `documents`:** the output carries both. Decide when `image_ids` goes.
- **Tell the backend teammate about the contract additions:**
  - `output.failure` (`EARLY_FAILURES`, `RETRYABLE_FAILURES` in `services/create/entry.py`)
  - `output.billable`
  - `output.documents`
  - the `image_ready` event (with `image_id`)
  - `ArtifactStore.put_thumbnail` / `get_thumbnail`
  - `RunState.gate`
  - TODO(backend): the safety-review limiter is per process, so several workers need a shared one.
- **Block lists:** the banned-word and celebrity/brand lists in `guardrails/rules.yaml` are empty until the owner supplies them.
- **Request cache** (§5.1): keyed on `(prompt, settings, seed, model)`; needs a document store.
- **`uv lock`:** `fastapi` and `uvicorn` were installed with `uv pip install`, so the lock file doesn't record the `server` group yet.

### M7. Measure the acceptance targets (§12): needs live runs

Only measurable with real calls, so ask first:
- wall time for 4 images ≤ 17 s
- time to first image ≤ 8 s
- free images/day ≥ 170
- refusing an unsafe request ≤ 3 s live
- 3 LLM calls per first pass, live

Tests already assert the call count and "a blocked run charges no credits" on mocks.

---

## 4. Measured facts (2026-11-20), so nobody re-measures them

| fact | value | source |
|---|---|---|
| Cloudflare `flux-1-schnell` with a seed | HTTP 400 / 5006 (rejected) | real call 2026-11-19 |
| Cloudflare `flux-2-klein-4b`, same seed twice | **different pixels** (seed not honoured) | `smoke_test.py --only cf-klein-seed` |
| Groq `qwen3.8-27b` reading the Critic's 768 px JPEG | works in JSON mode; ~1,330 prompt tokens | `smoke_test.py --only groq-vision` |
| OpenRouter `/images` response fields | `data[].b64_json`, `data[].media_type`, `usage.cost` | recorded fixture |
| gpt-image-2 `low`, 1:1 | $0.00596, 196 image tokens, 1024² PNG | same |
| OpenRouter catalogue vs spec | qwen-image-3 refs 3 (spec 4); seedream-5-0-flash n=1; seedream-5-0-lite sizes 2K/4K only; muse-image unserved; FLUX.2 has no `resolution` | catalogue GET, `.cache/openrouter_image_models.json` |
| Gate system prompt | 452 tokens (268 prose + 184 schema) | `o200k_base` |

---

## 5. Open questions for the owner

1. Run the two free Groq checks in M1?
2. Approve the Klein reproducibility check (~$0.028)?
3. Compiler placement (M3): compile per chain entry in the gateway, or once for the first entry?
4. Remove the deprecated `RunState` fields, and when should `image_ids` go?
5. The banned-word and celebrity/brand lists.
6. Track `execution-steps/` in git?
