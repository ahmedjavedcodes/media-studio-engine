# UI Pickers & Observability — Spec

## Current state of the three pickers
- **Image models** — deferred to batch (b). The catalog returns `models: []` on purpose; the
  picker isn't built. It also needs the `model` request field + the gateway starting the
  chain at the pick.
- **Reasoning models** — **blocked on item 7.** No `stacks` field yet: the R1–R4 slugs/prices
  aren't settled and R1 conflicts with CLAUDE.md. Can't show until you give the slugs/prices
  and R1 is fixed.
- **Resolution** — not built. The catalog returns `max_resolution`, but there's no control.
  Buildable now.
- **First:** fix F1 (shim `plan` → `auth`), or the Plan dropdown changes the catalog but not
  the run — a "pro" run still executes as free.

## Resolution picker (now)
- Add a Resolution select: 1K / 2K / 4K, capped at the plan's `max_resolution` (free/pro/max
  2K, ultra 4K).
- Size becomes **base ratio × resolution**, clamped to the chosen model's max side
  (flux-1-schnell is 1024). So a preset is now ratio + resolution, not a fixed pixel pair.
- Free default stays ≤1K on flux-schnell; higher resolution only on models that support it.

## Image-model picker (batch b)
- Render `catalog.models`: selectable when allowed, greyed with the upgrade plan when locked.
  Default "Auto". Send the chosen `id` back as `model`. Needs the gateway change from (b).

## Reasoning-stack picker (blocked — item 7)
- Build once the R-stack slugs/prices are confirmed and R1 is fixed. Free locked to R1; paid
  pick up to their tier. Send `stack`.

## Observability (now — most data already exists)
The Trace already records route decisions, each call's model/role, status and wall time;
RunState records spend (LLM calls, neurons, cost) and seeds; the Inspector polls it live.
Missing: persistence and a fuller view.

Build:
- **Persist recent runs.** Today Trace/RunState are in-memory in the shim. Store the last N
  (shim: SQLite or JSON on disk; real system: **LangSmith**, already the planned store of
  record). Key by run_id, timestamp, prompt.
- **Run-history list** — recent runs: prompt, status, model, cost, latency, time.
- **Run-detail view** — the whole pipeline for one run:
  - input: prompt, settings, plan, operation, N
  - route: gate verdict, child, operation, any Decider calls
  - per step: agent · model · tool called · tokens/neurons · cost · latency · decision/reasoning
  - output: images, scores + ranking, the document
  - totals: cost, wall time, LLM calls, seeds
- Keep it a **developer view** (the Inspector already is), separate from the product UI.
  Reasoning text stays dev-only — never shown to end users.
- Real system: wire **LangSmith** as the store of record and link each run to its trace.

## Order
1. Fix F1 (plan → auth) so the plan controls affect runs.
2. Resolution picker.
3. Observability: persist recent runs + the run-detail view.
4. Image-model picker with batch (b).
5. Reasoning picker once item 7 is unblocked.

## Needs you
- The R-stack model slugs + measured prices (unblocks the reasoning picker and item 7).
- Confirm observability's store: LangSmith for real, plus local disk persistence in the shim
  for now — or LangSmith only.
