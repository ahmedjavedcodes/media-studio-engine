# IMAGE-CREATE.md

Implementation spec for the **image creation harness**. Build exactly what is in this file.

---

## 1. Scope

Build a standalone Python package that turns a text prompt into one or more finished, quality-checked images through a small hierarchy of agents.

The harness is a **library**, not a service. It exposes one entry function, runs to completion, and returns a result envelope. It must be runnable and testable from a plain script with no web server, no database and no queue.

### In scope

- Complexity router
- Image Grand Agent (supervisor)
- Prompt Enhancer agent
- Image Creator (no LLM)
- Critic agent (vision)
- Model gateway with provider failover and a circuit breaker
- Run state with a plan object, event emission and cooperative cancellation
- Budget enforcement (tokens, cost, retries)

### Non-goals — do NOT build these

- Image editing, merging, poster/text layers, video
- The Grand Agent's multi-step planner (stub it, see §8)
- Session layer, agent registry, context assembler as separate services
- API server, auth, credit ledger, object storage backend, SSE transport, database
- Agentic/long-term memory

Anything not listed in "in scope" is out. If a requirement seems missing, implement the smallest stub behind a port (§6) and leave a `TODO(backend)` comment.

---

## 2. Stack

- Python 3.11+
- Pydantic v2 for every model and every agent output
- LangGraph for the fixed enhance → generate → critique sequence only
- Plain Python for the supervisor loop, routing, retries, budget and cancellation
- `httpx` (async) for provider calls
- LangSmith for tracing (optional, behind an env flag — never required to run)
- `pytest` + `pytest-asyncio`

Everything is `async`. No blocking I/O in agent code.

**Do not** use LangGraph for the supervisor loop, retries or branching. See §9.

---

## 3. Project layout

```
image_harness/
  __init__.py
  entry.py                 # create_images() — the only public entry point
  models/
    contracts.py           # Goal, StoppingCondition, AuthContext, Envelope, Status
    run_state.py           # RunState, PlanStep, Budget
    agent_io.py            # EnhancerOutput, CriticOutput, GenerationResult
  agents/
    base.py                # Agent base class
    router.py              # complexity classifier
    image_grand_agent.py   # supervisor (plain code loop)
    prompt_enhancer.py
    image_creator.py       # no LLM
    critic.py
  gateway/
    model_gateway.py       # image provider failover
    llm_gateway.py         # LLM calls by tier
    breaker.py             # circuit breaker
    providers/
      base.py
      cloudflare.py
      openrouter.py
  ports/
    run_store.py           # Protocol + InMemoryRunStore
    artifact_store.py      # Protocol + LocalArtifactStore
    event_sink.py          # Protocol + ListEventSink
  config/
    agents.yaml            # per-agent model policy by plan tier
    providers.yaml         # provider chain, timeouts, breaker thresholds
    presets.yaml
  prompts/
    prompt_enhancer.md
    critic.md
    router.md
tests/
scripts/
  run_once.py              # CLI smoke test
```

Agent system prompts live in `prompts/*.md` and are loaded at runtime. **Never hardcode a system prompt in a `.py` file.**

---

## 4. Core contracts

Every delegation passes `Goal` + `StoppingCondition` + `AuthContext`. Every agent returns an `Envelope`. No exceptions.

```python
# models/contracts.py

class Status(str, Enum):
    DONE = "done"
    DONE_WITH_ISSUES = "done_with_issues"
    NEEDS_INPUT = "needs_input"
    ESCALATE = "escalate"
    REJECTED = "rejected"          # safety block — terminal
    CANCELLED = "cancelled"
    FAILED = "failed"

class Goal(BaseModel):
    task: str
    instructions: str

class StoppingCondition(BaseModel):
    success: str
    max_tokens: int
    max_retries: int
    deadline_ms: int

class AuthContext(BaseModel):
    model_config = ConfigDict(frozen=True)   # REQUIRED: never mutable
    plan: Literal["free", "pro", "max", "ultra"]
    execution_mode: Literal["auto", "manual"]
    model_demand: Literal["cheap", "balanced", "best"]

class Spend(BaseModel):
    tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    retries_used: int = 0

class EnvelopeState(BaseModel):
    status: Status
    spend: Spend
    reason: str | None = None

class Envelope(BaseModel):
    output: dict[str, Any]       # references and scalars ONLY — never image bytes
    summary: str                 # <= 2 sentences, written by the agent's LLM
    state: EnvelopeState
```

**Hard rule:** `output` never carries image bytes or base64. Images are `image_id` strings resolved through the artifact store.

---

## 5. Run state and the plan

```python
# models/run_state.py

class PlanStep(BaseModel):
    step: int
    agent: str                        # internal name
    public_label: str                 # user-facing, e.g. "Creating your image"
    status: Literal["pending","running","done","failed","cancelled","skipped"]
    output_ref: str | None = None
    branches_total: int | None = None
    branches_done: int | None = None
    branches_aborted: int | None = None
    partial_outputs: list[str] = []
    spend: Spend = Spend()

class Budget(BaseModel):
    tokens_total: int
    cost_cap_usd: float
    reserve_fraction: float = 0.30    # held back for retries
    tokens_spent: int = 0
    cost_spent_usd: float = 0.0

    def remaining_for_steps(self) -> tuple[int, float]: ...
    def can_afford(self, est_tokens: int, est_cost: float) -> bool: ...

class RunState(BaseModel):
    run_id: str
    user_id: str
    status: Status | Literal["running"]
    auth: AuthContext
    budget: Budget
    plan: list[PlanStep]
    replans_used: int = 0
    cancel_requested: bool = False
    cancelled_at_step: int | None = None
    events: list[dict] = []
    created_at: datetime
```

`RunState` must serialise to plain JSON at every point. No live handles, no open connections, no non-serialisable fields. Add a test that asserts `RunState.model_dump_json()` round-trips after every step.

### Public labels

`public_label` is the only string that may reach a user. Internal agent names must never appear in events or output. Allowed labels: `"Understanding your request"`, `"Creating your image"`, `"Finishing up"`.

---

## 6. Ports (backend seams)

The backend is built by someone else. Define each dependency as a `Protocol` with a working local implementation.

```python
class RunStore(Protocol):
    async def save(self, state: RunState) -> None: ...
    async def load(self, run_id: str) -> RunState | None: ...

class ArtifactStore(Protocol):
    async def put(self, data: bytes, mime: str) -> str: ...   # returns image_id
    async def get(self, image_id: str) -> bytes: ...
    async def url(self, image_id: str) -> str: ...

class EventSink(Protocol):
    def emit(self, run_id: str, event: dict) -> None: ...      # sync, fire-and-forget
```

Ship `InMemoryRunStore`, `LocalArtifactStore` (writes to `./.artifacts/`) and `ListEventSink`. Nothing in `agents/` may import a concrete implementation — only the Protocol.

---

## 7. Agents

### 7.1 Base

```python
class Agent(ABC):
    name: str
    def __init__(self, config: AgentConfig, gateway: LLMGateway, events: EventSink): ...
    async def run(self, goal: Goal, stop: StoppingCondition,
                  auth: AuthContext, inputs: dict) -> Envelope: ...
```

One class, built from config. `AgentConfig` holds `name`, `prompt_file`, `model_policy`, `tools`, `output_model`.

Every agent's LLM output is parsed into its `output_model`. On validation failure: **one** repair attempt (hand the validation error back to the model), then return `Status.FAILED`. The repair attempt draws from the step's existing token budget — it does not extend it.

### 7.2 Complexity router

Small, fast model. One call, structured output.

```python
class RouteDecision(BaseModel):
    route: Literal["simple", "complex"]
```

`simple` → straight to the Image Grand Agent, no planning call. `complex` → the Grand Agent stub (§8).

### 7.3 Prompt Enhancer

```python
class EnhancerOutput(BaseModel):
    enhanced_prompt: str
    must_have: list[str] = Field(min_length=1, max_length=6)
    missing_info: list[str] = []      # triggers NEEDS_INPUT if non-empty
```

`must_have` items must be **derived only from the user's own words** — testable requirements, not the enhancer's additions. This is stated explicitly in `prompts/prompt_enhancer.md`.

If `missing_info` is non-empty, the Enhancer returns `Status.NEEDS_INPUT`. The Image Grand Agent tries to resolve it from preset defaults first (§8.3) before escalating.

### 7.4 Image Creator

**No LLM. Do not give this class a model.** It formats a request, calls the model gateway, stores the result, returns the ID.

```python
class GenerationResult(BaseModel):
    image_id: str
    provider: str
    model: str
    seed: int
    cost_usd: float
```

### 7.5 Critic

Vision model. Returns four fields in a single call:

```python
class CriticOutput(BaseModel):
    quality_score: int = Field(ge=0, le=10)
    safety_verdict: Literal["allow", "block"]
    safety_reason: str | None = None
    critique: str
```

Scores against `must_have` **and** the user's raw prompt — never against the enhanced prompt alone.

`safety_verdict == "block"` is terminal: the run ends with `Status.REJECTED` regardless of score, plan tier or remaining retries. It must not be reachable through any retry path. Write a test for this.

---

## 8. Image Grand Agent — the supervisor

This is the only complex component. **Its loop is plain Python.** The LLM is invoked at exactly two points: choosing a retry lever, and deciding whether to escalate. On a first-pass success it makes **zero** LLM calls.

### 8.1 Loop

```
build plan (static for creation: enhance → generate → critique)
for each step:
    if run_state.cancel_requested: freeze and return (§10)
    if not budget.can_afford(step estimate): return DONE_WITH_ISSUES with best attempt
    emit(step started)
    dispatch step
    subtract actual spend from budget
    emit(step finished)
    handle envelope status
```

### 8.2 Retry policy

- Triggered when `quality_score < threshold(plan)` for a single image, or when **every** branch fails for an N-image request.
- Max retries: `free=1`, all other plans `2`.
- Completed steps keep their outputs. A retry resumes at the failed step — it does not re-run the enhancer unless re-enhancing is the chosen lever.
- Retry levers, chosen by one LLM call given the critique:

| Lever | Allowed on |
|---|---|
| `reseed` | all plans |
| `re_enhance` | all plans |
| `switch_model` | pro and above |
| `raise_tier` | max and above |

- Out of retries → return the best attempt with `Status.DONE_WITH_ISSUES`, include every attempt in `output["image_ids"]` ordered best-first, and put the remaining critique in `output["issues"]`.

### 8.3 Resolving questions locally

On `NEEDS_INPUT` from the Enhancer, try preset defaults first (size, aspect ratio, style). Only escalate what the defaults cannot answer. In `execution_mode == "auto"`, never escalate — always take the default.

The harness **does not talk to the user**. `NEEDS_INPUT` is returned upward in the envelope; the caller handles it.

### 8.4 N variations

One shared enhancement, then N parallel `creator → critic` branches with different seeds, via `asyncio.gather(..., return_exceptions=True)`.

- Every branch's image is returned regardless of score. The Critic **ranks** them; best first.
- A replan is triggered only if every branch fails.
- One branch raising an exception must not cancel the others. Return the successful images, mark the failed branch in the plan step, and record it for refund.
- Cap concurrent branches at `min(n, MAX_PARALLEL_BRANCHES)` from config. Default 4.

### 8.5 Grand Agent stub

Create `agents/grand_agent.py` with a stub that accepts a `complex` route, returns `Status.ESCALATE` with reason `"multi-step planning not implemented"`, and nothing else. Do not implement planning.

---

## 9. Model gateway

Plain code. No LLM. The Image Creator calls `generate(prompt, size, tier, seed)` and never learns a failover happened.

- Provider chain from `config/providers.yaml`: Cloudflare Workers AI first, then paid fallback.
- Per-provider circuit breaker: trip after N consecutive failures, skip for a cooldown window, probe periodically to detect recovery. **This is required, not optional** — free-tier-first makes failover the normal path, and without a breaker every request pays a timeout before falling back.
- Per-call timeout from the stopping condition's `deadline_ms`.
- Handle these distinctly:

| Failure | Gateway behaviour |
|---|---|
| Rate limit | Next provider |
| Provider down / timeout | Trip breaker, next provider |
| Content refusal | **Raise `ContentRefused`** — returned up to the Image Grand Agent as a real decision |
| All providers exhausted | Raise `NoProviderAvailable` |

`llm_gateway.py` does the same for text/vision models, selecting the model by agent and plan tier from `config/agents.yaml`.

---

## 10. Events and cancellation

### Events

`emit(event)` is passed down with the goal. Agents call it at step boundaries.

It is **fire-and-forget and must never raise**. Wrap the body in `try/except Exception: pass`. A failed event write must never break generation. Write a test that proves a raising `EventSink` does not fail a run.

Event shape: `{"run_id", "step", "public_label", "phase", "ts"}` where `phase ∈ {started, progress, finished}`. Internal agent names must not appear.

### Cancellation

Cooperative, via `run_state.cancel_requested`. Check it:

1. at every step boundary, and
2. **immediately before any image generation call** — this is the one that matters, because that call is where the money is.

In-flight provider requests get an `httpx` cancel/abort.

On cancel, **freeze** the plan — do not clear it:

- completed steps keep `output_ref` and spend
- the in-flight step gets `status="cancelled"` plus `branches_done`, `branches_aborted`, `partial_outputs`
- pending steps get `status="skipped"` — never removed
- set `cancelled_at_step`
- report only actual spend in the envelope so the caller can release the rest

**Partial images from branches whose Critic never ran have not been safety-checked.** Either run the Critic's safety pass on them before including them in `output`, or exclude them. Cancellation must not bypass the safety verdict. Write a test for this.

---

## 11. Config

`config/agents.yaml` — per agent: prompt file, required capabilities, and a model per plan tier.

```yaml
critic:
  prompt: prompts/critic.md
  requires: [vision, structured_output]
  models:
    free: <small vision model>
    pro: <better vision model>
    max: <best vision model>
```

Thresholds, retry caps, `MAX_PARALLEL_BRANCHES` and breaker settings all live in config. **No magic numbers in code.**

---

## 12. Build order

Build and test each milestone before starting the next.

1. **M1** — models, ports with local implementations, config loading. Test: `RunState` JSON round-trip.
2. **M2** — `llm_gateway` + `Agent` base with Pydantic validation and the repair path. Test: malformed output repairs once, then fails cleanly.
3. **M3** — Prompt Enhancer. Test: `must_have` is populated; `missing_info` produces `NEEDS_INPUT`.
4. **M4** — `model_gateway` + breaker + one real provider. Test with a fake provider: failover, breaker trips and recovers, `ContentRefused` propagates.
5. **M5** — Image Creator + Critic. Test: `block` verdict ends the run as `REJECTED`.
6. **M6** — Image Grand Agent happy path, single image. Test: supervisor makes zero LLM calls on first-pass success.
7. **M7** — retries and levers. Test: cap respected; completed steps reused; `DONE_WITH_ISSUES` returned when exhausted.
8. **M8** — N parallel branches, ranking, partial-failure handling.
9. **M9** — events and cancellation, including the frozen plan and the safety rule on partial outputs.
10. **M10** — `scripts/run_once.py` end to end.

---

## 13. Acceptance criteria

The build is done when all of these pass:

- [ ] `python scripts/run_once.py "a cozy coffee shop"` returns an image id and writes a file to `./.artifacts/`
- [ ] A first-pass success makes exactly three LLM calls total: router, enhancer, critic
- [ ] The Image Grand Agent makes zero LLM calls when nothing fails
- [ ] `safety_verdict: block` ends the run as `REJECTED` and no retry path can override it
- [ ] Retry cap is enforced per plan; exhaustion returns `DONE_WITH_ISSUES` with all attempts, best first
- [ ] A 4-image request runs 4 branches concurrently and returns all 4, ranked
- [ ] One failing branch does not cancel the other three
- [ ] An `EventSink` that raises on every call does not fail a run
- [ ] Cancelling mid-run leaves a plan where completed steps keep outputs, the in-flight step is `cancelled`, and pending steps are `skipped`
- [ ] Unchecked partial images are never returned in `output`
- [ ] `RunState` serialises to JSON and reloads at every step boundary
- [ ] Budget overrun is refused **before** a generation call, never after
- [ ] No system prompt is hardcoded in a `.py` file
- [ ] No internal agent name appears in any emitted event

---

## 14. Rules for the implementer

1. Ask before adding a dependency not listed in §2.
2. Do not build anything in the non-goals list, even if it seems necessary. Stub it behind a port and add `TODO(backend)`.
3. The supervisor loop is plain Python. If you find yourself giving the Image Grand Agent an LLM call per step, stop — that is the design error this spec exists to prevent.
4. Never put image bytes in an agent message or envelope.
5. `AuthContext` is frozen. If code needs to change it, the design is wrong.
6. Safety is a verdict, not a score. They never share a field and are never compared.
7. Write the test for each milestone before moving on. Mock providers and LLMs — no test may make a real paid API call.
