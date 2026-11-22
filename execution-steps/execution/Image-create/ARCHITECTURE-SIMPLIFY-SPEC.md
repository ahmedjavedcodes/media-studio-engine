# Architecture Simplify — Spec-Driven Document

For Claude Code to execute. Reference plan: `project-status/ARCHITECTURE-SIMPLIFY.md`.
**Stop before wiring Jev and wait for approval. Do not commit or push unless asked.**

---

## 1. Problem Statement

The image-side agent structure is too complex to hold in your head. Today `agents/grand/`
is a 24-line ESCALATE stub, while `agents/image_grand/` is not a grand agent at all — it
is the create pipeline runner. The real grand-agent duties (code checks, Gate, routing,
hand-over) are buried in a function in `entry.py`. Routing is an LLM field, not code.
There is no place for edit, merge or restyle to attach.

Replace this with one simple, linear hierarchy. **Move and rename code only. Do not
rewrite logic. The create pipeline runs exactly as it does today.**

Target:

```
create_images()                          thin public entry
 └─ ImageGrandAgent                       Safety Gate first, then Jev routes to a child
     ├─ ImageGenerationAgent              Jev routes the operation → create (only one now)
     │    └─ image-create-agent (REAL)    plain code: brief → image-gen model → critic
     │         ├─ brief            LLM, structured fields
     │         ├─ image-gen-model  TOOL: the image model, no reasoning
     │         └─ critic           LLM vision: safety + score + rank
     │    (edit · merge · restyle          stubs, built later)
     └─ DesignerLayoutAgent               stub, built later
```

Routing uses **Jev (TypeSafe AI)** at both levels: the Grand Agent picks the child, the
Generation Agent picks the operation. The LLM plan for complex requests is future work,
added together with the layout agent.

---

## 2. Functional Requirements

- **ImageGrandAgent** runs the Safety Gate first on every path (code checks → Gate →
  reviewer if "review" → block = REJECTED). It then uses Jev to route to one of its two
  children: Image Generation Agent or Designer & Layout Agent.
- **ImageGenerationAgent** uses Jev to route the operation: create / edit / merge /
  restyle. Only create is wired now; the rest are stubs that return ESCALATE.
- **image-create-agent** is real and runs plain code: brief → image-gen model (tool) →
  critic. No LLM plan now.
- The LLM plan path (for complex requests) is added later, with the layout agent.
- The Brief always runs; it is never skipped. A detailed prompt is still prose, so the
  Brief turns it into structured fields. When the prompt is already detailed, the Brief
  structures it and fills only real gaps — it does not pad it or add objects.
  `prompt_is_detailed` is recorded but no longer skips the Brief.
- System prompts live in `prompts/<agent>/`, one folder per agent (gate, brief, critic
  now; edit, merge, restyle, designer as placeholders). Prompt content and the LangSmith
  names stay the same — only the folder layout and the loader paths change.
- Routing shortcut: when the operation is explicit (button or slash command), route by
  plain code with no call. Only ambiguous free text goes to Jev.
- The Gate still produces and stores `route` and `task_type`, but nothing reads them.
- Missing details pass up as `NEEDS_INPUT` to the caller, which triggers the user hook.
  No agent below the Grand Agent talks to the user.
- Every operation returns a document with layers.

**Build order** (each step ends green on ruff, mypy, pytest):

1. Rename-only moves: `git mv` modules into the new layout, rename classes, fix imports,
   delete empty `agents/brief/` and `agents/edit/`.
2. Extract the Safety Gate out of `entry.py` into `ImageGrandAgent`.
3. Add `ImageGenerationAgent` (operation routing, explicit → code) with create wired and
   edit / merge / restyle stubs. Add the `DesignerLayoutAgent` stub as the Grand Agent's
   second child, with the Grand Agent routing generation vs design.
4. Make the Brief always run: remove the `prompt_is_detailed` skip so every create runs
   Gate → Brief → Critic. Keep the field in the schema, unread.
5. Stop reading the Gate's `route` / `task_type`. Delete `agents/grand/`.
6. Reorganize `prompts/` into per-agent folders (`prompts/gate/`, `prompts/brief/`,
   `prompts/critic/`, placeholders for the rest). Update the loader's paths only. Keep
   prompt content and the LangSmith names unchanged.
7. **STOP for approval** → wire Jev at both levels behind a `Decider` protocol, with safe
   fallbacks (see §5). Update the one CLAUDE.md router line.
8. Rewrite `ARCHITECTURE.md` sections 1–4; remove dead yaml entries.

---

## 3. API Contracts (inputs, outputs, data shapes)

- **Entry:** `create_images()` keeps its current signature and Envelope output. Add an
  optional `operation` kwarg (`create|edit|merge|restyle`, default inferred). Defaults
  reproduce today's behaviour.
- **Delegation:** unchanged — `run(goal, stop, auth, inputs) -> Envelope`. Depth is at
  most Grand → Generation → Create.
- **Gate:** `GateDecision` schema unchanged: `{safety, route, task_type,
  prompt_is_detailed}`. `route`, `task_type` and `prompt_is_detailed` are stored on
  RunState, read by nothing (the Brief now always runs).
- **Router (Jev):** a `Decider` protocol — ambiguous input in, a routing choice out.
  `JevDecider` is the first implementation, used at both the Grand Agent (child choice)
  and the Generation Agent (operation choice).
- **Output:** Envelope with `output["documents"]` (one image layer each, ranked). Reuse
  the existing `Status` values; do not add a new one.
- **No schema changes** anywhere (`schemas/*`, `contracts/envelope.py`).

---

## 4. Constraints

- Move and rename only. The only new logic is the two routers and the stubs.
- Keep these names exactly (stored in RunState JSON, trace roles, config): config keys
  `prompt_enhancer`, `image_creator`, `gate`, `critic`; the LangSmith prompt names;
  `PlanStep.agent` strings. Python class and module names change, and prompt files move
  into per-agent folders (loader paths updated) — but their content and LangSmith names
  do not change.
- **Do not touch:** `gateway/*` (chains, providers, fallback, budget, compiler,
  registry), `schemas/*`, `contracts/envelope.py`, `harness/run_state.py`, `trace.py`,
  `ports/*`, `guardrails/*` logic, `presets/`; the Gate, Brief and Critic prompt content
  and logic, settle, cancel, the budget guard, seeds, thumbnails. (The only create-flow
  change is removing the `prompt_is_detailed` skip so the Brief always runs.)
- **Out of scope** (separate tasks): per-plan image caps (free 2, paid 4); user-picked
  image or reasoning models; the LLM plan path and the Designer agent's real behaviour.
- CLAUDE.md wins on any conflict. Update only its one router-fallback line.
- No commit or push unless asked.

---

## 5. Edge Cases and Error Handling

- Jev down, unkeyed, timeout or error → safe fallback: at the Grand Agent, default to the
  Image Generation Agent; at the Generation Agent, default to create. No LLM fallback.
- Jev is early access and unused in the repo today. Build it behind the `Decider`
  protocol, verify the request format first, and test with a fake (mocked HTTP). No real
  Jev call outside `playground/` until verified.
- A non-create operation (edit / merge / restyle) → its stub returns ESCALATE with its
  own reason. Ambiguous text classified as one of these while they are stubs → run as
  create and log it. One config flag flips this when the real operations exist.
- Complex requests have no LLM plan yet → handled when the layout agent is added.
- Gate `block` → REJECTED, no credits charged.
- `NEEDS_INPUT` passes up unchanged to the user hook.
- The UI's fixed "escalate" text is wrong for edit/merge stubs. Set a per-stub reason;
  the frontend text change is out of scope — flag it for the frontend owner.
- 25 uncommitted files exist. Commit or stash first, then work on a new branch.

---

## 6. Acceptance Criteria

- The Safety Gate runs first on every path.
- The Grand Agent routes to one of its two children; the Generation Agent routes to an
  operation. Explicit operations route in plain code with zero model calls.
- create routes to image-create-agent and runs plain code: brief → image-gen model →
  critic.
- Stubs (edit, merge, restyle, Designer) return ESCALATE, each with its own reason.
- create's Envelope output shape is unchanged (pinned by a test).
- The Brief runs on every create, including already-detailed prompts; `prompt_is_detailed`
  is no longer read.
- System prompts live under `prompts/<agent>/`; prompt content and LangSmith names are
  unchanged, only paths moved.
- The Gate's `route` and `task_type` are no longer read anywhere.
- `agents/grand/` is deleted; empty `agents/brief/` and `agents/edit/` are removed.
- A first-pass create is still exactly 3 LLM calls (Gate, Brief, Critic).
- `ruff`, `mypy` and `pytest` are green at every step; pass count ≥ the recorded
  baseline.
- No file under "Do not touch" is changed.
