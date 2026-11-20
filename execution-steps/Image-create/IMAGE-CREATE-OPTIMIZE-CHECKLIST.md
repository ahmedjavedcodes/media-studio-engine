# IMAGE-CREATE-OPTIMIZE (v2.2) — checklist

Status as of 2026-11-20, branch `image-create`, all pushed:
`22bfdd0` linear create, seeds, the Gate and review round 2; `a984c5b` settings screening, Critic in-step retry, settle rule, D4, D5, media type, thumbnails, document output, streaming.
**Everything still open below is carried into [IMAGE-CREATE-OPTIMIZE-2.md](IMAGE-CREATE-OPTIMIZE-2.md) for the next session.**
Paid spend: **$0.0060 of the $5 cap** (ledger: `.cache/paid_spend.jsonl`).
Tests: 617 passed, 3 skipped (agents with no prompt); ruff, mypy, tsc and next build clean. `fastapi` + `uvicorn` installed 2026-11-20, so the dev-server tests now run.

Legend: `[x]` done · `[~]` partly done · `[ ]` not started

## Build order (§13)

- [~] **1. D1: send the seed**
  - [x] OpenRouter: seed sent only to models that take one (registry-driven)
  - [x] Test: outbound request body contains `seed` on a seed-capable model
  - [x] Derive the seed from `(run_id, round)` + branch offset (`derive_seed`); tests in `test_seeds.py`
  - [x] Free check run (2026-11-20): Cloudflare `flux-2-klein-4b` does NOT reproduce (same seed 42 twice, different pixels); recorded `honours_seed=False`
  - [ ] Same check on OpenRouter FLUX.2 Klein (paid, ~$0.028, needs approval): the only candidate left for reproducible evals
  - [x] Free tier has no reproducible model: `flux-1-schnell` rejects `seed` (HTTP 400 / 5006), Klein 4B ignores it
- [x] **2. Agents on Groq, a different model each**
  - [x] Gate `gpt-oss-20b`, Brief/enhancer `gpt-oss-120b`, Critic `qwen3.8-27b`; Cloudflare kept as fallback
  - [x] `safety_review` chain (`gpt-oss-safeguard-20b`) used only by the safety reviewer, only on a `review` verdict
  - [x] qwen3.8-27b vision verified with the Critic's request shape (~1,330 tokens per 768 px image)
  - [x] CLAUDE.md updated; production chain order and fallback path tested
- [~] **3. Step 0 code checks, then the Gate (§4.1)**
  - [x] Code checks (minimal, owner's choice): prompt length, control characters, injection phrases (NFKC + whitespace folding); `guardrails/rules.yaml`
  - [x] Review round 2: invisible format characters (Unicode Cf: zero-width, bidi, soft hyphen) stripped before matching; regex misses ("ignore the instructions above", "your previous", singular) and false positives ("disregard the instructions in the manual", "show the instructions on a box") fixed and tested
  - [x] `settings` keys and values screened at step 0 and shown to the Gate and the reviewer (they reach the enhancer's prompt too)
  - [ ] Banned-word and celebrity/brand lists: slots exist, EMPTY until the owner supplies them
  - [ ] Request cache (needs the document store, §4.3)
  - [x] Gate replaces the router: `safety`, `safety_categories`, `route`, `task_type`, `prompt_is_detailed` (`gpt-oss-20b`)
  - [x] `review` → safety reviewer on the `safety_review` chain; `block` → REJECTED before any image
  - [x] Fails closed: an undecided Gate or review ends the run FAILED, never a default allow
  - [x] Verdict recorded once on `RunState.gate`; travels with complex requests; supervisor refuses a non-allow verdict
  - [x] Blocked / refused run: REJECTED with zero image spend
  - [x] Every early exit carries `output["failure"]` (`code_check_refused`, `safety_block`, `invalid_input`, `gate_undecided`, `review_undecided`, `review_rate_limited`); `RETRYABLE_FAILURES` marks the outages
  - [x] `output["billable"]` on every envelope: False before any image step (incl. escalation), True once an image was generated
  - [x] Safety reviewer behind a process-wide limiter (3 per 60 s, waits up to 5 s), fails closed as `review_rate_limited`; per process only (TODO(backend): shared limiter for several workers)
  - [x] `task_type` edit/merge intentionally ignored by create (no uploads, services not built): comment, trace decision, test
  - [ ] `prompt_is_detailed` is recorded but not acted on (it will skip the Brief once the Brief exists)
  - [ ] Gate prompt measured at 452 tokens (268 prose + 184 injected schema, `o200k_base`), 2.3x the §8 budget of 200; the schema alone nearly fills it, so meeting it means provider structured output instead of an in-prompt schema
- [x] **4. Document output (§4.3)**: every delivery also returns `documents` (one image layer each, ranked, version 1); `image_ids` kept alongside until the backend moves over
- [x] **5. Thumbnails and per-call instrumentation**
  - [x] Trace records `usage.cost`, seed sent, seed honoured
  - [x] Adapter reads `media_type` per image (pinned by a real recorded response)
  - [x] Provider's `media_type` passed through; the stored type comes from the bytes, a mismatch is logged
  - [x] 512 px and 256 px thumbnails at store time (`put_thumbnail` / `get_thumbnail`), swept with their image; the original lookup can never return a thumbnail
  - [ ] The Critic still downscales to 768 px itself; reading a thumbnail instead waits on E1
- [x] **6. Stream each image to the UI as it finishes**: `image_ready` event (with `image_id`) the moment the Critic allows an image; never for a blocked or unchecked one; the frontend shows them under the stepper
- [ ] **7. Critic**
  - [ ] One call ranks all N thumbnails
  - [ ] Four-dimension rubric with scale anchors; safety a separate binary field
  - [ ] Rubric loaded by `task_type`
  - [ ] Critique text only below threshold
  - [ ] E1 picks the thumbnail size
- [ ] **8. Brief and Compiler**
  - [ ] Brief returns structured fields (§4.2), temperature 0.2, max_tokens 150, grounding rule
  - [ ] Skip the Brief when `prompt_is_detailed`
  - [ ] `gateway/compiler.py`: per-model templates, word budgets, banned vocabulary, aspect-ratio defaults, blocklist re-run
- [~] **9. OpenRouter registry**
  - [x] 15 models with the §6.3 fields, checked against the live catalogue
  - [x] Tiers: entry, pro, max, ultra (all paid)
  - [x] One registry per process, built from one cache; manual refresh (sync script)
  - [x] Plan guard: model tier and price-multiplying settings checked in the gateway; unregistered paid models refused
  - [x] `n` routed to a same-tier model that can return that many, else a typed error
  - [~] `usage.cost` into the budget: recorded and metered daily; `estimate_image_usd` exists but is NOT ENFORCING
- [~] **10. Settle rule, guard collapse, budget fixes**
  - [x] Settle (§5.5): a blocked image is dropped and the rest delivered (with an issue); every image blocked → REJECTED + `supervisor.gate_miss` log and trace decision
  - [x] D5: one guard per step, taking the branch count (one decision; unaffordable branches refused before any call); cancel still checked per branch
  - [x] Budget reserve 0.30 -> 0.10 (review round 2) -> 0.0: nothing re-runs a step any more, so nothing draws on a reserve
  - [ ] D3: replace wrong estimates (neurons ~260 vs 57.6; flat $0.04 per image under-estimates ultra models; critique estimate 12,000 tokens vs ~1,350 measured, which keeps free n=2 fragile)
  - [ ] Wire the USD estimate into the gateway and check it against a real `usage.cost`
- [ ] **11. Run E2–E7**, then set tier models and Critic thresholds

## v2.2: no regeneration loop in create

- [x] Retry tests sorted into keep (~13) and delete (~30); report given
- [x] Remove retry, replan and lever-picking from create (graph fully linear; Critic re-ask kept: same images, no new spend)
- [x] Lever prompts, `LeverChoice` schema, `max_retries` and `levers` config removed; supervisor makes no LLM call
- [x] Kept tests rewritten onto the linear path (`test_linear.py`); `test_retries.py` deleted; `test_config`, `test_trace`, `test_branches` updated
- [x] Inspector: retry counter, lever columns, re-ask counter and reserve marker removed
- [x] Retry planner no longer shares the Brief's Groq model (it makes no call)
- [x] RunState retry fields (`retries_used`, `replans_used`, `lever`, `feedback`) marked DEPRECATED in comments; not removed
- [ ] Decide whether to drop those fields from the RunState contract (always 0 / None now; backend teammate affected)
- [x] Critic re-ask removed: the graph runs exactly once; the Critic retries ONCE inside its own step (`params.retries`), never after a block or an unfixable failure; `max_critic_reasks` removed, `critic_reasks_used` deprecated

## Defects (§9)

- [~] D1: seed (derived from run id now; reproducibility check still needs a real call)
- [x] D2: input safety now runs first (Gate); the Critic keeps the OUTPUT safety verdict, as §5.4 requires
- [ ] D3: budget estimates wrong
- [x] D4: enhancer temperature 0.2 (per-agent `temperature` in agents.yaml, sent with every call; tested)
- [x] D5: one guard per step (a 4-image run logs 3 guard decisions, was 9)

## Prompts and reasoning (§8)

- [x] User text framed as data, not instructions (already in every agent prompt)
- [ ] Declared token budgets with a failing test (Gate ≤ 200, Brief ≤ 600, Critic ≤ 150)
- [ ] `reasoning_effort: low` for Gate, Brief, Critic, with a test that reasoning tokens stay at 0
- [ ] At most 2 few-shot examples; schema not described in prose

## Experiments (§11)

- [ ] E1: Critic thumbnail size (needs approval)
- [ ] **E2: prompt vs model quality** (approved, ~$0.16; the compiled-prompt arm needs the Compiler)
- [ ] E3: gpt-image-2 low/medium/high vs 3 models (needs approval; `high` arm ~$6.30 needs separate approval)
- [ ] E4: ranking critic validation (needs approval)
- [ ] E5: Critic calibration (needs approval)
- [ ] E6: `prompt_is_detailed` precision (needs approval)
- [ ] E7: `n_per_call` vs parallel requests (needs approval)

## Acceptance targets (§12) — none measured yet

- [~] Refusing an unsafe request ≤ 10 neurons / ≤ 3 s (0 neurons by design: the Gate runs on Groq; time not measured live)
- [~] 3 LLM calls per first-pass run (gate, enhancer, critic: asserted in mocked tests; not measured live)
- [ ] Wall time for 4 images ≤ 17 s
- [ ] Time to first image ≤ 8 s
- [ ] Free images/day ≥ 170
- [ ] Critic score spread ≥ 2 points on a mixed set
- [ ] Seeds honoured 100% on seed-capable models
- [x] Blocked run charges no credits (`billable: false` on every pre-image exit; the backend must honour it)
- [ ] Compiled prompt length within each model's budget

## Open questions for the owner

- [x] Install `fastapi` + `uvicorn`: done with `uv pip install` (`uv.lock` NOT updated; run `uv lock` to record the `server` group)
- [ ] Drop the always-0/None retry fields from `RunState` (`retries_used`, `replans_used`, `lever`, `feedback`)? Backend teammate affected.
- [ ] Supply the banned-word and celebrity/brand lists for the code checks.
- [ ] Approve the paid reproducibility check on OpenRouter FLUX.2 Klein (~$0.028).
- [ ] Document output is ADDITIVE for now (`documents` alongside `image_ids`): confirm, and when may `image_ids` go?
- [ ] Backend contract additions to share with the teammate: `output.failure`, `output.billable`, `output.documents`, the `image_ready` event (with `image_id`), and `ArtifactStore.put_thumbnail` / `get_thumbnail`
- [x] Review findings on `review_bundle_linear_gate.md`: applied (round 2)
- [x] Critic re-ask: done as recommended (deletions + one retry inside the Critic)
- [x] Screen `settings` values: done
- [ ] Gate prompt budget: switch to provider structured output to drop the in-prompt schema?
- [ ] Run `uv lock` to record the `server` group in the lock file
