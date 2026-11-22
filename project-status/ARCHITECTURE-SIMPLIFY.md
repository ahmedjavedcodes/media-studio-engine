# ARCHITECTURE-SIMPLIFY.md (plan only, no code changed)

Plan to replace the image-side agent structure with the simpler hierarchy. Status: **awaiting review**.
Written from a full read of `src/media_engine/`, `CLAUDE.md`, `project-status/ARCHITECTURE.md` and
`execution-steps/Image-create/IMAGE-CREATE-OPTIMIZE.md` (v2.2).

Paths you named do not exist: there is no `claude/` folder and no `SETUP.md` in the repo. I used the files
above instead, and put this plan next to `ARCHITECTURE.md`.

Decisions you already made (in answer to my questions):

| # | Decision |
|---|---|
| D-1 | Plan lives in `project-status/`. |
| D-2 | `GateDecision` (schema), the gate prompt and `GateRecord` stay untouched. The Gate's `route` and `task_type` are still produced and stored on RunState, but nothing reads them any more. |
| D-3 | The operation router has **no LLM fallback**. Jev classifies ambiguous text; if Jev is unavailable, the router falls back to `create`. |
| D-4 | "Ad / complex" is an **explicit request field** only (`kind="ad"`). No guessing from free text. |

---

## 1. Current structure

What is actually in the code today (the branch is `image-create`, create is already linear per v2.2):

```
create_images()                        services/create/entry.py (337 lines)
 |  does the job of the Grand Agent AND the router, but is a function, not an agent
 |-- [0] code checks                   guardrails/code_checks.py (no model)
 |-- [1] Gate         LLM  fast_llm    -> {safety, route, task_type, prompt_is_detailed}
 |        review  -> SafetyReviewer     safety_review chain, 3 req/min, fails closed
 |-- block -> REJECTED
 |-- route == "complex"  -> GrandAgent stub     agents/grand/  => ESCALATE
 |-- route == "simple"   -> ImageGrandAgent     agents/image_grand/supervisor.py (1226 lines)
 |                            LangGraph, linear:  enhance -> generate -> critique
 |                              PromptEnhancer   LLM  reasoning   (the "Brief")
 |                              ImageCreator     NO LLM, but an Agent subclass
 |                              Critic           LLM  vision   (one call ranks all N)
 |                            + budget/cancel guard, N-branch fan-out, settle, documents
 |-- task_type != "create"   ignored on purpose (review round 2): handled as a new image
```

First-pass success is 3 LLM calls (Gate, Brief, Critic). The supervisor makes none.

**Where it is more complex than the target**

1. **Two "grand" packages with opposite meanings.** `agents/grand/` is a 24-line ESCALATE stub. `agents/image_grand/` is
   not a grand agent at all: it is the *create pipeline runner* (enhance, generate, critique, settle). The real
   grand-agent duties (code checks, Gate, review, hand-over) are buried in a function in `entry.py`.
2. **Routing is an LLM field, not code.** `GateDecision.route` and `.task_type` come out of the safety call. Nothing is
   routed by plain code, and `task_type` is computed and then deliberately ignored.
3. **No operation layer.** There is no place for edit, merge or restyle to attach. `services/edit/`, `services/merge/`,
   `agents/edit/`, `agents/brief/` are empty scaffolds.
4. **`ImageCreator` is an `Agent` subclass with no reasoning.** It carries `AgentConfig`, a goal and a stopping
   condition for a function that formats a request and calls the gateway.
5. **Dead scaffolding from the retry loop that v2.2 removed:** the `supervise` step kind (estimate and deadline), the
   `image_grand_agent` and `grand_agent` yaml entries (chains that are never used), `supervisor_spend`,
   `retries_used`, `replans_used`, `lever`, `feedback`. Some of it is a stored-state contract, so I do not touch it here.
6. **`project-status/ARCHITECTURE.md` is stale.** It still describes a router LLM call, the lever-choice supervisor
   call and a retry loop. The code has no router (the Gate replaced it), no retries and no lever call.

**LLM calls that could be plain code:** only the Gate's `route` and `task_type` outputs (a few output tokens inside a
call that has to run anyway). The create path is already at the minimum of 3 calls. This change does not cut LLM
calls; it removes layers and names.

---

## 2. Target structure

```
create_images()                              thin public entry (signature kept, 2 optional kwargs added)
 +-- ImageGrandAgent                         agents/image_grand/grand.py
      |-- Safety Gate (always first)           code checks -> Gate -> SafetyReviewer if "review" -> block = REJECTED
      |-- child 1: ImageGenerationAgent        agents/image_generation/agent.py
      |     |-- OperationRouter
      |     |     explicit operation (button / slash command)  -> plain code, no call
      |     |     ambiguous free text -> Decider (Jev) ; Jev unavailable -> "create"
      |     |-- create   -> ImageCreateAgent   (REAL)
      |     |     path by `kind`:
      |     |       plain (default) -> today's linear pipeline, no supervisor LLM
      |     |       "ad"            -> STUB, returns ESCALATE (later: supervisor plans effects/scene,
      |     |                          Brief still the only field writer, Designer agent)
      |     |     parts:  Brief (LLM, temp 0.2)
      |     |             ImageGenModel (a TOOL: Compile + image gateway, no reasoning)
      |     |             Critic (LLM vision: safety verdict, score, ranking, one call for all N)
      |     |-- edit     -> stub   |  merge -> stub  |  restyle -> stub
      +-- child 2: DesignerLayoutAgent         stub (built later)
```

Rules carried into the new structure:

- Every delegation is still `run(goal, stop, auth, inputs) -> Envelope` (the existing contract). Delegation depth is
  at most Grand, then Generation, then Create.
- Missing details: the create pipeline already returns `NEEDS_INPUT` (manual mode). It passes up unchanged through
  Generation to Grand to the caller, which triggers the user hook. No agent below the Grand talks to the user.
- Memory: RunState only. Nothing added.
- Documents: create already returns `output["documents"]` (one image layer each, ranked). Unchanged.
- Create stays linear: one generation, all N images returned, the user picks. No retry code is added.
- The pipeline steps stay exactly as they are. Mapping to the 7 steps in IMAGE-CREATE-OPTIMIZE: steps 0 and 1 (code
  checks, Gate) move up to the Grand Agent; steps 2 to 6 (Brief, Compile, Generate, Critic, Settle) stay in create.

---

## 3. The diff

| Today | Mark | Becomes | Notes |
|---|---|---|---|
| `create_images()` body (checks, Gate, review, route, hand-over) | **MERGE** | `ImageGrandAgent` | Logic moved, not rewritten. `create_images()` stays as the thin public entry. |
| `agents/gate/` (Gate, SafetyReviewer, limiter) | **KEEP** (moved under `image_grand/`) | same code | Gate stays one call. Not merged with the Brief. |
| Gate's `route` / `task_type` | **KEEP** (schema), stop reading | recorded only | D-2. |
| `agents/grand/` (ESCALATE stub) | **MERGE** | the `ad` path stub in the create agent | Package removed. Constants move. |
| `agents/image_grand/supervisor.py` (`ImageGrandAgent`) | **RENAME** | `ImageCreateAgent` | Same code. Gains one small `kind` switch at the top of `run()`. |
| `agents/image_grand/attempt_graph.py` | **KEEP** (moved) | `image_generation/create/graph.py` | Unchanged. |
| `agents/prompt/enhancer.py` (`PromptEnhancer`) | **RENAME** | `Brief` | Class and module only. See risk R-5 for what keeps the old name. |
| `agents/generation/image_creator.py` (`ImageCreator`) | **RENAME** | `ImageGenModel`, a tool, in `tools/` | Same `run()` contract; stops being an `Agent` subclass. |
| `agents/critic/critic.py` | **KEEP** (moved) | `image_generation/create/critic.py` | Unchanged. |
| (none) | **ADD** | `ImageGenerationAgent` | Router plus dispatch. |
| (none) | **ADD** | `OperationRouter` | Explicit routes in code; ambiguous goes to the Decider. |
| (none) | **ADD** | `Decider` Protocol and `JevDecider` | CLAUDE.md names the interface. First use of the installed `typesafe-sdk`. |
| (none) | **ADD** | edit, merge, restyle stubs | Return `ESCALATE` with a per-operation reason. |
| `agents/designer/` (empty) | **ADD** (stub) | `DesignerLayoutAgent` in `image_grand/designer.py` | Placeholder only. |
| `agents/edit/`, `agents/brief/` (empty packages) | **REMOVE** | | Superseded by the new layout. |
| `agent` entries `image_grand_agent`, `grand_agent` in agents.yaml | **RENAME** / **REMOVE** | `image_create_agent` / gone | Dead chains. |
| `services/edit`, `merge`, `merge_text`, `product_ads`, `product_design`, `memory/`, `presets/` | **KEEP** | | Not image-agent hierarchy; untouched. |
| gateway, providers, schemas, compiler, budget, ports, guardrails, harness/run_state, trace | **KEEP** | | Out of scope. |

---

## 4. File-by-file change list

New layout (all under `src/media_engine/`):

```
agents/image_grand/
    grand.py              ADD     ImageGrandAgent (code checks, Gate, review, delegate)
    gate.py               MOVE    from agents/gate/gate.py        (+ limiter.py)
    designer.py           ADD     DesignerLayoutAgent stub
agents/image_generation/
    agent.py              ADD     ImageGenerationAgent
    router.py             ADD     OperationRouter, Operation = create|edit|merge|restyle
    decider.py            ADD     Decider Protocol, JevDecider
    edit.py merge.py restyle.py   ADD  stubs
    create/
        agent.py          MOVE+RENAME  from agents/image_grand/supervisor.py -> ImageCreateAgent
        graph.py          MOVE         from agents/image_grand/attempt_graph.py
        brief.py          MOVE+RENAME  from agents/prompt/enhancer.py -> Brief
        critic.py         MOVE         from agents/critic/critic.py
tools/
    image_gen_model.py    MOVE+RENAME  from agents/generation/image_creator.py -> ImageGenModel
```

Edited in place:

| File | Change |
|---|---|
| `services/create/entry.py` | Shrinks to: validate args, build the gateway/ports, call `ImageGrandAgent`. Adds optional `operation` and `kind` kwargs. Everything else it does moves to `grand.py`. |
| `services/create/config/agents.yaml` | Rename `image_grand_agent` to `image_create_agent`, remove `grand_agent`. Keys `gate`, `safety_reviewer`, `prompt_enhancer`, `critic`, `image_creator` stay (R-5). |
| `services/create/config/harness.yaml` | Only step 7 below (per-plan image cap), and only if you approve it. |
| `services/create/loader.py` | Only if step 7 is approved (a `max_images` map by plan). |
| `project-status/ARCHITECTURE.md` | Rewrite sections 1 to 4 to match the code and this hierarchy (last step). |
| `scripts/run_once.py`, `scripts/dev_server.py`, `playground/*` | Import path updates only. |

Tests (import and name updates only, no behaviour edits, except the new tests listed under step 3 to 6):
`test_entry`, `test_gate`, `test_supervisor`, `test_linear`, `test_branches`, `test_seeds`, `test_safety_block`,
`test_critic`, `test_prompt_enhancer`, `test_image_creator`, `test_config`, `test_prompt_budgets`, `test_trace`,
`tests/conftest.py`.

**Untouched:** `gateway/*` (chains, providers, fallback, budget, compiler, registry), `schemas/agent_io.py`,
`contracts/envelope.py`, `harness/run_state.py` and `trace.py`, `ports/*`, `guardrails/*`, all prompts under
`prompts/`, `presets/`, the Critic/Brief/Gate logic, the settle rule, cancel, budget guard, seeds, thumbnails.

---

## 5. Stub now vs built later

| Piece | Now | Later |
|---|---|---|
| Safety Gate, code checks, reviewer | real (moved) | |
| ImageGenerationAgent, explicit routing | **real** | |
| create, plain path (Brief, ImageGenModel, Critic) | **real** (moved) | |
| Jev classification of free text | **real, small** (step 5) | Needs the Jev API verified first (R-2). |
| create, `kind="ad"` supervisor path | stub, `ESCALATE` | supervisor plans effects/scene, Brief writes fields, Designer composes layers |
| edit, merge, restyle | stub, `ESCALATE` | real services, plus long-term memory (edit and merge) |
| DesignerLayoutAgent | stub | text and layout layers |
| User-picked image model, user-picked reasoning model on paid | **not built** | See R-6. |

---

## 6. Risks and what could break

| # | Risk | Mitigation |
|---|---|---|
| R-0 | The working tree has 25 uncommitted/staged files (M7 to M10, playground, docs). A refactor on top of that is hard to review or revert. | Step 0: you commit or stash first; refactor on a new branch. |
| R-1 | **Two behaviour changes that follow from your decisions, not just moves.** (a) Today a free-text prompt the Gate calls `complex` returns ESCALATE. With D-2/D-4 nothing reads `route`, so it now just generates a plain image. (b) Today `task_type` edit/merge is ignored and the request is created (a deliberate review-round-2 decision: text-only "change the background to blue" is still served). If the router sends ambiguous text to the edit stub, that request now ends in ESCALATE instead of an image. | For (b) I recommend: while edit/merge/restyle are stubs, an **ambiguous** classification is logged and runs as create; only an **explicit** operation reaches a stub. A single config flag flips this when the real ones exist. Needs your OK (Open decision O-1). |
| R-2 | Jev is early access and nothing in the repo uses `typesafe-sdk` yet (0.7.5 is installed; `Choice`/`AsyncTypeSafeClient` exist). I have not verified request format, latency or failure modes. CLAUDE.md also says to ask before adding providers. | Step 5 is gated on your go-ahead. Behind the `Decider` Protocol, with a fake in tests (respx for the HTTP). Any Jev error, timeout or empty key falls back to `create`. |
| R-3 | **Frontend message.** The UI shows a fixed text for `escalate` ("needs more than one step"). That is wrong for an edit/merge stub. `Status` is a shared contract, so I do not add a new value. | Per-stub `summary`/`reason` is set; the frontend text change is out of scope. Flagged for the frontend owner. |
| R-4 | Large mechanical diff: about 14 test files change imports, plus `git mv` of 5 modules. A bad rename could hide a real change. | Step 1 is rename-only with zero logic edits, committed on its own, tests green before and after. No re-export shims. |
| R-5 | Renaming config keys and prompt files (`prompt_enhancer`, `image_creator`) would also change `PlanStep.agent` strings stored in RunState JSON, trace roles and `test_config`/`test_prompt_budgets`. | Keep those config keys, prompt filenames and stored strings in this change. Only the Python class and module names change. A later cosmetic rename can follow. |
| R-6 | "User picks the image model in the prompt bar" and "reasoning model fixed on free, selectable on paid" are not in the code today (no request field for a chosen model). Building them touches the gateway, which is out of scope. | Left out. Needs its own task. Tell me if you meant it to be part of this one. |
| R-7 | Free tier max 2 images vs max 4 on paid. The code has one flat `max_images_per_request: 4`. Existing budgets (free 600 neurons) may already refuse n=2. | Separate, optional step 7 (config only). It is a behaviour change, not structure. Credits scale on the backend side, nothing to do here. |
| R-8 | CLAUDE.md says the `Decider` falls back to an LLM with a Pydantic schema when the Jev key is empty. D-3 chose "no LLM fallback". CLAUDE.md wins on conflict. | Open decision O-2. |
| R-9 | `create_images()` is the backend seam (ARCHITECTURE.md section 8) and `dev_server` calls it. | Signature and Envelope output stay identical; the two new kwargs are optional with today's behaviour as default. A test pins the unchanged output shape. |

---

## 7. Ordered steps

Each step ends green on `ruff`, `mypy` and `pytest` (`PYTHONPATH=src .venv/bin/python -m pytest -q`). I will not
commit or push unless you ask.

0. **Baseline.** You commit or stash the current work. New branch. Run the full suite and record the pass count.
1. **Rename-only moves.** `git mv` the modules to the layout in section 4, rename the classes, fix imports. No
   logic edits. Remove the empty `agents/brief/`, `agents/edit/`. Tests unchanged except imports.
2. **Extract the Safety Gate.** Move steps 0 and 1 (checks, Gate, review, block, trace, spend roll-up) out of
   `entry.py` into `ImageGrandAgent`. `entry.py` becomes thin. `test_entry` and `test_gate` must pass untouched.
3. **ImageGenerationAgent and explicit routing.** Add `OperationRouter` (explicit only) and the edit/merge/restyle
   stubs. `create_images(..., operation=None)`. New tests: each explicit operation routes in plain code with zero
   model calls; stubs return ESCALATE with their own reason; unchanged Envelope for create.
4. **Create path switch.** `kind="plain"` (default) runs today's pipeline; `kind="ad"` returns the ESCALATE stub (the old
   GrandAgent constants). Stop reading the Gate's `route` and `task_type`. Delete `agents/grand/`. Tests: `kind="ad"`
   escalates after the Gate; the Gate still runs first on every path; `NEEDS_INPUT` passes up unchanged.
5. **Decider and Jev (needs your go-ahead).** `Decider` Protocol, `JevDecider`, wire the ambiguous branch. Fall back
   to `create` on any failure. Tests use a fake Decider and mocked HTTP; no real call outside `playground/`.
6. **Designer stub.** `DesignerLayoutAgent` placeholder plus the grand's second child slot. Decide how it is reached
   (Open decision O-3).
7. **Optional: per-plan image cap** (free 2, paid 4) in `harness.yaml` and `loader.py`, with tests.
8. **Docs and cleanup.** Rewrite `ARCHITECTURE.md` sections 1 to 4 (it is stale today). Remove dead yaml entries.
   Leave `supervise`, `retries_used` and the other stored-state leftovers for a later task.

---

## 8. Open decisions for you

- **O-1.** Ambiguous text classified as edit/merge/restyle while those are stubs: run as create and log (my
  recommendation, keeps the review-round-2 behaviour), or end in ESCALATE?
- **O-2.** CLAUDE.md says the Decider falls back to an LLM when the Jev key is empty; you chose no LLM fallback. Do
  you want CLAUDE.md updated to match, or the LLM fallback kept on a chain that respects the one-Groq-model rule?
- **O-3.** How does the Grand Agent reach the Designer child? I suggest an explicit request field (`target="design"`),
  with the stub returning ESCALATE. Without a trigger the stub is unreachable dead code.
- **O-4.** Confirm removing `agents/grand/`: is there a planned cross-media Grand Agent above the Image Grand, or is
  the Image Grand the top for now?
- **O-5.** Include step 7 (per-plan image cap) in this change, or split it out?
- **O-6.** User-picked image model and reasoning model (R-6): confirm they are out of scope here.
