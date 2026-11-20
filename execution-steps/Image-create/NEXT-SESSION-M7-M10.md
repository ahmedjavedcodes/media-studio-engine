# NEXT-SESSION-M7-M10.md

Handoff for the session that builds **M7 → M10** of the image-creation harness. Written 2026-11-19 at the end of M6. Read this, then `IMAGE-CREATE.md` (the spec, sections 8–13), then the code. The spec is the authority; this file records what was built, what was decided, and what is still open.

## 0. Start here

Paste this to begin:

> Read execution-steps/NEXT-SESSION-M7-M10.md, execution-steps/IMAGE-CREATE.md and your memory notes. We're on branch image-create with M1–M6 done (last commit 600e040). Ask me the open questions in §9 in one batch, then start M7. Build and test one milestone at a time; commit and push only when I say so.

## 1. State at handoff

- Branch `image-create`, HEAD `600e040` (M6), pushed. Working tree clean apart from `execution-steps/` (untracked on purpose: it holds the spec and this file).
- Tests: **374 passed, 3 skipped** (`PYTHONPATH=src .venv/bin/python -m pytest -q`). ruff and mypy are clean. The 3 skips are prompt-file checks for agents with no prompt yet (`router`, `image_grand_agent`, `image_creator`).
- Commits: M1 `11ac979`, M2 `2ea8241`, M3 `50cc9d5`, M4 `33203ea`, M5 `9d8dc82`, M6 `600e040`.
- **M6 was committed before any reviewer pass.** Expect M6 review findings from the user; apply them with M7.
- **No real model call has ever been made for M4–M6.** `.env` is on another machine. The user will add it after the image-create work is finished. Everything is tested against fakes (respx, `ScriptedChat`, `ScriptedImages`, `FakeProvider`).

### What exists (all under `src/media_engine/`)

| Area | Files | Notes |
|---|---|---|
| Contracts | `contracts/envelope.py` | `Goal`, `StoppingCondition`, `AuthContext` (frozen), `Spend` (has `neurons`, `llm_calls`, `.plus()`), `Envelope` (rejects bytes/base64 in `output`), `Status` |
| Run state | `harness/run_state.py` | `RunState`, `PlanStep` (has `output` dict, added in M6), `Budget` (tokens, USD, neurons; `reserve_fraction`; `can_afford(..., use_reserve=)`; `charge()`) |
| Config | `services/create/loader.py`, `config/harness.yaml`, `config/agents.yaml`, `presets/library/image_create.yaml` | `policy.stopping_condition(kind, plan, success)`, `policy.budget_for(plan)`, `policy.levers_for(plan)`, `quality_threshold`, `max_retries`, `levers`, `max_parallel_branches`, `max_images_per_request`, `max_seed`, `aspect_ratio_tolerance`, `step_estimates`, `deadlines_ms` (incl. a `supervise` kind) |
| Agent base | `agents/base.py` | `call_structured` with ONE repair, `Structured.failure` (a `FailureKind`), `Agent.failed(reason, spend, failure)`, `gateway_failure()` |
| Agents | `agents/prompt/enhancer.py`, `agents/generation/image_creator.py`, `agents/critic/critic.py` | see §4 for input/output shapes |
| Supervisor | `agents/image_grand/attempt_graph.py`, `agents/image_grand/supervisor.py` | M6. One attempt only. `TODO(M7)`, `TODO(M8)`, `TODO(M9)` markers show the seams |
| Gateway | `gateway/router.py` (`Gateway`), `breaker.py`, `model_gateway.py`, `llm_gateway.py`, `fallback.py`, `errors.py`, `chains.yaml`, `providers.yaml` | M4. `ModelGateway.generate(prompt, *, width, height, tier, seed, deadline_ms)` |
| Ports | `ports/` | `RunStore`/`InMemoryRunStore`, `ArtifactStore`/`LocalArtifactStore` (`mark_orphaned`, `sweep_orphaned`), `EventSink`/`ListEventSink`, `safe_emit`, `make_event` |
| Schemas | `schemas/agent_io.py` | `RouteDecision`, `EnhancerOutput`, `MissingItem`, `GenerationResult`, `CriticOutput`, `FailureKind` |
| Tests | `tests/unit/harness/` (`test_supervisor.py`, `test_critic.py`, `test_image_creator.py`, `test_safety_block.py`, `fakes.py`, ...), `tests/unit/gateway/` | `fakes.py` has `ScriptedChat`, `ScriptedImages`, `RaisingSink`, `image_bytes()`; `test_supervisor.py` has the `build()`, `create()` and `RecordingStore` helpers to reuse |

### What does NOT exist yet

- The **complexity router** agent: no `router.md` prompt, no `Router` class. `RouteDecision` (schema) and the `router` entry in `agents.yaml` exist.
- `prompts/image_create/image_grand_agent.md` (the lever-choice prompt) and a lever-choice output schema.
- The **Grand Agent stub** (`agents/grand/` is an empty package; spec §8.5).
- `entry.py` (`create_images()`) and `scripts/run_once.py` (the `scripts/` folder does not exist).
- Any retry loop, any `n > 1` (`SupervisorInputs.n` is `Literal[1]`), in-flight cancellation.

## 2. Hard rules (from CLAUDE.md, the spec and the user)

- **Never read, print, log or commit `.env`.** Never create it. If `.env` is missing, smoke scripts print "missing key" and make no calls; that is expected.
- **No paid providers**, and never set `ALLOW_PAID=true`, without the user's explicit permission. **Ask before any real call**, even a free Cloudflare/Groq one.
- Tests mock every provider. Real calls only in `playground/` and `evals/`.
- Ask before adding a dependency or a provider. Agent code never calls a provider directly: everything goes through `gateway/`.
- No system prompt in a `.py` file (a test scans for it). Prompts live in `src/media_engine/prompts/**.md`. When you add `router.md` and `image_grand_agent.md`, remove the names from `PENDING_PROMPTS` in `tests/unit/harness/test_prompt_rules.py`.
- `output` in an Envelope never carries image bytes. `public_label` is the only string a user may see: the only labels are `"Understanding your request"`, `"Creating your image"`, `"Finishing up"`. **Do not add a label without asking.**
- `AuthContext` is frozen. Safety is a verdict, never a score.
- Do not build the backend (API, auth, DB, queue, payments). Stub behind a port with `TODO(backend)`.
- **Commit and push only when the user asks.** Stage files by name (never `git add -A`: `execution-steps/` must stay untracked). Use the attribution lines the harness gives you for commits.

## 3. Decisions already made (do not re-litigate)

- Code lives in `src/media_engine/` and reuses the existing `gateway/`; the spec's `image_harness/` layout is only a component list.
- **LangGraph** runs only the fixed enhance → generate → critique sequence, linear, state `{run_id}` only. The user's three constraints:
  1. `RunState` is the single source of truth; nothing about plan, budget, spend or status is mirrored into the graph state.
  2. The retry loop **wraps** the graph. No edge from critique back to enhance. One graph run = one attempt.
  3. Cancel, budget refusal, safety block and step failure all leave through the single END edge; no node raises past the graph; the step guard (`_begin`) is the only place those checks happen.
- Blocked images: mark orphaned with reason `"safety_block"` and let the sweep delete them.
- Aspect mismatch (flux-1-schnell is sent no width/height): **ignore it (log only) when a free provider served the image; report it in `issues` when a paid provider did.**
- `block` ends the run `REJECTED` whatever the score, plan or retries left. A block must never reach a retry path. A Critic failure means the image is unchecked: never deliver it.
- **Lever rule:** when a step failed with `failure == "content_refused"`, the ONLY permitted lever is `re_enhance`. `reseed` resends the same prompt to the same filter; `switch_model` and `raise_tier` would route around a safety decision.
- LLM calls are counted by summing `Spend.llm_calls` over the plan, never on the gateway (it is stateless).
- The Creator's `is_cancelled` input is NOT passed by the supervisor; the guard is the one place cancel is checked (`TODO(M9)` to move the hook off `inputs`).
- Unchecked images (generated, then cancelled or failed before a verdict) are marked orphaned (`"unchecked"`) and excluded.

## 4. Step-agent contracts (what the supervisor calls)

- **Enhancer** `inputs`: `prompt`, `settings` (dict), `feedback` (str, for `re_enhance`). Output: `enhanced_prompt`, `must_have`, `must_have_dropped`, `missing_info` (`[{kind, question}]`), `text_wanted`, `user_prompt`. `NEEDS_INPUT` when `missing_info` is non-empty.
- **Creator** `inputs`: `prompt`, `width`, `height`, `seed` (required), `tier` (optional plan override: the `raise_tier` hook), `is_cancelled`. Output: `GenerationResult` (`image_id`, `provider`, `model`, `seed`, `seed_honoured`, `cost_usd`, `neurons`, `width`, `height`, `mime`). Failures: `FAILED` with `output["failure"]`.
- **Critic** `inputs`: `image_id`, `prompt` (the user's RAW prompt), `must_have`. `DONE` → `{image_id, quality_score, critique, safety_verdict}`. `REJECTED` → `{blocked_image_id, safety_verdict, safety_reason}`. `FAILED` → unchecked image.
- `FailureKind`: `invalid_input`, `invalid_output`, `content_refused`, `no_provider`, `deadline`, `bad_request`, `gateway_error`, `bad_image`, `storage`, `internal`.

## 5. Unverified, blocked on `.env`

Smoke checks already exist in `playground/smoke_test.py` (free Cloudflare calls): `--only cf-vision-scout`, `cf-vision-gemma`, `cf-seed`. **Ask the user before running them**; they cannot run until `.env` is restored.

1. **The Critic's `image_url` message format on Cloudflare** and the two vision model IDs (`@cf/meta/llama-4-scout-17b-16e-instruct`, `@cf/google/gemma-3-12b-it`). If the format is wrong, every critique fails and there is no fallback. M6–M10 all rest on the Critic working.
2. **`seed_honoured`.** `ModelCard.honours_seed` is `None` for every model. The `reseed` lever is only meaningful once flux-1-schnell is shown to honour the seed. Until then treat `reseed` as unproven and **tell the user before M7 relies on it.**
3. Cloudflare refusal code `3030` and the refusal wording regex are from third-party reports only.
4. The `critique` neuron estimate (200) and the other `harness.yaml` numbers (budgets, thresholds, deadlines) are placeholders.

Build against the mocked Critic as before; keep these items marked UNVERIFIED in code comments and in the final report.

## 6. M7 — retries and levers (spec §8.2)

**Tests required:** cap respected per plan (`free=1`, others `2`); completed steps reused; `DONE_WITH_ISSUES` returned when retries are exhausted. Also: a block never reaches a retry path; `content_refused` permits only `re_enhance`; a retry draws on the budget reserve.

What to build:
- The retry loop in `ImageGrandAgent.run()` around `run_attempt()` (`TODO(M7)` markers in `supervisor.py`). Trigger: `quality_score < quality_threshold[plan]`. Cap: `policy.max_retries[plan]`. Levers allowed per plan: `policy.levers_for(plan)`. `retries_used` goes in `Spend`.
- **The one LLM call that chooses a lever**, given the critique: agent `image_grand_agent` (chain `reasoning`, already in `agents.yaml`). Needs `prompts/image_create/image_grand_agent.md` and a structured output model (e.g. `LeverChoice{lever, reason}` in `schemas/agent_io.py`), run through `Agent.call_structured` so it gets the repair and the typed failure. The call must be charged to a plan step so `run_spend()` still sums every LLM call. The first-pass path must still make **zero** supervisor LLM calls.
- Levers: `reseed` (new seed), `re_enhance` (rerun the Enhancer with `feedback` = the critique), `switch_model` (pro+), `raise_tier` (max+: the Creator already accepts `tier`).
- **Retries may spend the budget reserve:** use `Budget.can_afford(..., use_reserve=True)`. First-pass steps use `use_reserve=False` (the current guard). Budget refusal after at least one delivered attempt returns `DONE_WITH_ISSUES` with the best attempt (spec §8.1); with none it is `FAILED` (current M6 behaviour).
- Out of retries → `DONE_WITH_ISSUES`, `output["image_ids"]` holds **every attempt, best first**, `output["issues"]` holds the remaining critique.

Design problems to solve (and record as deviations/decisions in the review bundle):
1. **Resume at the failed step.** The graph always starts at `enhance`, but spec says a retry "resumes at the failed step" and keeps completed outputs. Suggested mechanism, keeping `RunState` the only source of truth: a retry sets the steps to redo back to `pending` and leaves the others `done`; `_begin` returns `None` (no events, no cost) for a step that is already `done`, so the same graph run passes straight through it. Check this against constraint 2.
2. **Attempt history.** Three `PlanStep`s cannot hold N attempts. Suggested: a JSON-safe attempts list on `RunState` (image_id, seed, score, critique, lever). It is a new field beyond the spec; document it like `PlanStep.output`.
3. **`switch_model`.** `ModelGateway.generate` takes only a tier. It needs a way to start at the next chain entry (e.g. skip the first). On the free path that is flux-1-schnell → flux-2-klein-4b. Note `agents.yaml` maps every plan to the same `image_create` chain, so `raise_tier` currently changes nothing; flag this to the user rather than silently inventing tier chains.
4. **Which failures trigger a retry** (the spec only names low score and "every branch fails"). Recommendation (needs the user's OK): Critic `FAILED` → retry the critique only on the existing image (that is "resume at the failed step"); Creator `deadline` or `bad_image` → one `reseed`; `no_provider` / `bad_request` / `invalid_input` → fail at once; `content_refused` → `re_enhance` only.
5. **Block after an earlier delivered attempt:** the run ends `REJECTED`. Decide whether earlier allowed images are returned (suggest: no, `image_ids` empty, to keep "block is terminal" simple) and ask the user.
6. **"Deciding whether to escalate"** is the spec's second LLM point. The spec gives no behaviour for it in M7; propose a minimal version or defer, and say so.

## 7. M8 — N parallel branches (spec §8.4)

**Tests required:** all N images returned, ranked best first; one failing branch does not cancel the others; a replan only if every branch fails; concurrency capped at `min(n, max_parallel_branches)` (default 4).

- One shared enhancement, then N `creator → critic` branches with different seeds via `asyncio.gather(..., return_exceptions=True)`.
- Every branch's image is returned regardless of score; the Critic **ranks**. A failed branch is marked on the plan step (`branches_total`, `branches_done`, `branches_aborted`, `partial_outputs` already exist on `PlanStep`) and **recorded for refund**. `RunState.replans_used` exists.
- Budget and cancel checks happen per branch, immediately before each branch's generation call. Keep one place to audit (the guard).
- `SupervisorInputs.n` is `Literal[1]`; widen it to `1..max_images_per_request`.
- A branch that gets `REJECTED` (block): the spec says block is terminal. Decide whether the whole run is `REJECTED` or only that image is dropped. **Ask the user.** Never deliver the blocked image either way.

**Biggest design question (decide with the user early):** the graph is linear with one generate node and one critique node, but N branches need concurrent `creator → critic` pairs. Options: (a) keep the 3-node graph and run `gather` inside the generate and critique nodes (generate-all then critique-all: simple, but more unchecked images in flight if cancelled); (b) run the graph's enhance once, then one graph run per branch with enhance already `done` (reuses the M7 "skip done steps" mechanism, one branch = one attempt, closest to the spec's branch model). Recommendation: (b), if the M7 mechanism works; confirm it does not weaken constraints 1–3.

## 8. M9 — events and cancellation (spec §10), and M10 — end to end (spec §12, §13)

**M9 tests required:** a raising `EventSink` never fails a run (also with branches); cancel mid-run leaves completed steps with outputs, the in-flight step `cancelled` (with `branches_done`, `branches_aborted`, `partial_outputs`), pending steps `skipped`, `cancelled_at_step` set; only actual spend is reported; **unchecked partial images are never returned.**

- Boundary cancel and the unchecked-image exclusion already exist from M6. M9 adds in-flight cancel: abort the in-flight provider call (cancelling the task aborts the httpx request) and freeze the plan.
- **Who sets `cancel_requested`?** Today the guard reads the in-memory `ctx.state`. In real use the backend sets it on its own copy via the `RunStore`. Decide how the harness sees it (e.g. a watcher that polls `RunStore.load(run_id)` during a step, or an explicit cancel handle returned by `create_images`). Move the Creator's `is_cancelled` callable off `inputs` (a callable cannot round-trip through JSON, and `RunState` must). Ask the user which shape they prefer.
- Partial branch images with no Critic verdict: exclude and `mark_orphaned` (already agreed), do not run a safety pass on them.
- Event shape is fixed: `{run_id, step, public_label, phase, ts}`, `phase ∈ {started, progress, finished}`. `progress` is not emitted yet. No internal agent name may appear in events or output.

**M10 build list:**
- The **router** agent (`RouteDecision`, a new `prompts/image_create/router.md`, chain `fast_llm`): `simple` → straight to the supervisor; `complex` → the Grand Agent stub in `agents/grand/` that returns `Status.ESCALATE` with reason `"multi-step planning not implemented"` and nothing else.
- `entry.py` with `create_images()`, the only public entry point (suggest `services/create/entry.py`; confirm the location). It builds `Gateway` (ONE instance per process: breaker and availability state live on it), `LLMGateway`, ports and the supervisor, and runs to completion.
- `scripts/run_once.py "a cozy coffee shop"`: prints the image id and leaves a file in `./.artifacts/`. Never print keys. LangSmith stays optional behind its env flag.
- A fully mocked end-to-end test (respx for HTTP) proving: first-pass success = **exactly 3 LLM calls** (router, enhancer, critic) with **zero** from the supervisor, and a file in the artifact store. A real run needs `.env`: ask first.

## 9. Open questions to ask the user in one batch before M7

1. M7: OK with the recommendation in §6 item 4 for which failures trigger which retry?
2. M7: if a later attempt is blocked, are earlier allowed images returned or not (suggest: not)?
3. M7: `reseed` is unproven until `cf-seed` runs and `raise_tier` changes nothing with today's `agents.yaml`. Build them anyway behind config, or wait? (Suggest: build, mark UNVERIFIED, report.)
4. M7: how much of "deciding whether to escalate" should be built?
5. M8: whole-run `REJECTED` or drop-that-image when one of N is blocked?
6. M8/M9: branch structure (a) or (b) in §7, and how cancel reaches the harness (§8).
7. M10: location of `entry.py`; and should M10's real run wait for `.env`?
8. Any M6 review findings to apply first?

## 10. Acceptance checklist (spec §13): status

| Criterion | Status |
|---|---|
| `run_once.py "a cozy coffee shop"` returns an id and writes to `./.artifacts/` | M10 |
| First pass = exactly 3 LLM calls (router, enhancer, critic) | supervisor part done (zero calls, M6); router + entry in M10 |
| Supervisor makes zero LLM calls when nothing fails | **done** (M6); M7 must keep it |
| `block` → `REJECTED`, no retry path can override | **done** at Critic and run level (M5/M6); M7/M8 must keep it |
| Retry cap enforced per plan; exhaustion → `DONE_WITH_ISSUES`, all attempts best first | M7 |
| 4-image request runs 4 branches concurrently, ranked | M8 |
| One failing branch does not cancel the others | M8 |
| Raising `EventSink` does not fail a run | **done** (M6); re-test in M8/M9 |
| Cancel mid-run: completed keep outputs, in-flight `cancelled`, pending `skipped` | boundary version done (M6); in-flight in M9 |
| Unchecked partial images never returned | boundary version done (M6); branches in M9 |
| `RunState` serialises and reloads at every step boundary | **done** (M6); extend to retries/branches |
| Budget overrun refused before a generation call | **done** for one attempt (M6); extend to retries (reserve) and branches |
| No system prompt hardcoded in a `.py` file | test exists; keep green when adding prompts |
| No internal agent name in any event | **done** (M6); keep for new steps |

## 11. Working loop and pitfalls

- Run tools through the venv interpreter: `PYTHONPATH=src .venv/bin/python -m pytest -q`, `MYPYPATH=src .venv/bin/python -m mypy src/media_engine`, `.venv/bin/ruff check src tests`, `.venv/bin/ruff format`. `uv` is not on the shell PATH. `gh` is in `~/.local/bin` (`export PATH="$HOME/.local/bin:$PATH"`); `git push` works.
- **zsh does not word-split an unquoted `$VAR`.** For multi-file scripts (e.g. mutation checks) use `bash <<'BASH'` with arrays, check the backups exist, and `cmp` after every restore. This once left seven mutations in the source.
- Per milestone: write the key rule's test first, build, run lint + mypy + tests, then **mutation-check** the safety rules (break each rule on purpose, confirm tests fail, restore from backup, `cmp`).
- When the user says **"summary for reviewer"**: write one Markdown bundle in the scratchpad (context header, design decisions to scrutinise with questions, full source of new/changed files, **every relevant test file**, relevant config, the git diff), scan it for secrets, and hand over the path. The reviewer sees only the bundle, so leave nothing out. The user pastes findings back: apply the valid ones, push back with evidence on wrong ones.
- Tell the user plainly when something is unverified, skipped, or a deviation from the spec; record deviations in the bundle.
