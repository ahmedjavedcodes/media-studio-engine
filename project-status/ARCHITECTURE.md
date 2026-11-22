# Architecture: image creation

Read this before building anything that talks to the harness. Sections 1 to 6 are the AI side
(this repo, `src/media_engine`). Sections 7 and 8 are the seam to the backend and the contract the
frontend is built against. The harness is a **library**: no server, no database, no queue.

Legend: `->` flows to, `=>` returns, `[ ]` a step or component, `( )` a decision, `{ }` data.

---

## 1. End to end

```
client
  -> create_images(prompt, auth, n, settings, run_id, run_store, artifacts, events)
       -> validate prompt and n                    -- bad input => Envelope FAILED(invalid_input), 0 model calls
       -> [router]  1 LLM call, chain fast_llm     -- its own failure => treated as "simple"
            -> route = ?
                 |
                 +-- simple  -> [Image Grand Agent]  (the supervisor, sections 2 to 4)
                 |                 => Envelope: DONE | DONE_WITH_ISSUES | NEEDS_INPUT |
                 |                              REJECTED | CANCELLED | FAILED
                 |
                 +-- complex -> [Grand Agent stub]   no LLM, no planning
                                   => Envelope: ESCALATE, reason "multi-step planning not implemented"
       -> router spend is added to the Envelope's spend
  => Envelope { output, summary, state { status, spend, reason } }
client
```

First-pass success makes **exactly 3 LLM calls**: router, enhancer, critic. The supervisor itself
makes none (its only LLM call is the retry-lever choice, section 3).

`output` never carries image bytes or base64. Images are `image_id` strings, resolved through the
ArtifactStore to a URL.

---

## 2. The attempt graph (LangGraph, linear, one run = one attempt)

```
START -> [enhance] -> [generate] -> [critique] -> END
          LLM call     NO LLM       LLM call
          (fast_llm)   (image       (vision model:
                       gateway)     score + safety)

- no conditional edges, no edge from critique back to enhance
- graph state is { run_id } only; the real state is RunState (the single source of truth)
- every early exit (cancel, budget refused, block, step failed) is recorded on the run context
  by the step guard; the remaining nodes pass through, so there is ONE way out: END
- generate is the only step that spends image money: the guard checks cancel and budget
  immediately before the paid call
```

Public labels (the ONLY strings a user may see; internal agent names never leave RunState):

```
enhance  -> "Understanding your request"
generate -> "Creating your image"
critique -> "Finishing up"
```

---

## 3. The supervisor loop (retries wrap the graph)

```
run(inputs)
  -> build RunState (3 plan steps pending, attempt 1 open)
  -> LOOP:
       -> [attempt graph]  START -> enhance -> generate -> critique -> END     (section 2)
       -> verdict?
            |
            +-- block (safety)  -------------------------------------> REJECTED   (terminal, never retried)
            |
            +-- cancel seen  ----------------------------------------> CANCELLED  (frozen plan)
            |
            +-- budget refused  -------> had a checked image? --yes--> DONE_WITH_ISSUES (best attempt)
            |                                                 --no---> FAILED
            |
            +-- step failed -> which failure?
            |       Critic failed (glitch)   -> critique again on the SAME image  (own cap)  -> LOOP
            |       Creator deadline/bad_image -> reseed                                   -> LOOP
            |       Creator content_refused  -> re_enhance ONLY                            -> LOOP
            |       anything else            -> had a checked image? yes: DONE_WITH_ISSUES
            |                                                         no:  FAILED
            |
            +-- ok, score < threshold (one image only)
            |       retries left? --no--> DONE_WITH_ISSUES  (every attempt, best first, + critique)
            |                     --yes-> [lever choice] ONE LLM call (the supervisor's only one)
            |                                lever in { reseed, re_enhance, switch_model, raise_tier }
            |                                -> reset ONLY the steps to redo to pending   -> LOOP
            |                                   (done steps are reused: no event, no cost)
            |
            +-- ok, score >= threshold  ------------------------------> DONE
  -> settle: rank checked images best first; orphan every stored image that is not delivered
  => Envelope
```

```
retry caps (per plan)      free 1, pro/max/ultra 2         (harness.yaml max_retries)
levers (per plan)          reseed, re_enhance: all   switch_model: pro+   raise_tier: max+
                           (raise_tier is hidden while every tier shares one chain: a no-op)
after content_refused      ONLY re_enhance (reseed re-sends to the same filter; the others route around it)
retries may spend          the budget reserve (30% of each meter)
```

---

## 4. N images (1 to 4)

```
[enhance] once  -> shared prompt
   |
   -> gather( generate x N )     seeds base+0 .. base+N-1, concurrency = min(N, max_parallel_branches)
   |     each branch: gate (cancel, budget) -> Image Creator -> store -> record
   |     a failed branch is marked (branches_aborted), recorded for refund; the others go on
   |
   -> gather( critique x N )     each stored image is scored and safety-checked
   |
   -> rank best first (ties: earlier branch)
   -> settle
        any block            -> REJECTED, NO images (the whole run, allowed siblings are orphaned)
        every branch failed  -> replan: ONE retry, same rules as section 3
        otherwise            -> every checked image returned, whatever its score
                                 (DONE if all N made it and the best passes, else DONE_WITH_ISSUES)
```

A low score never retries an N-image request. A replan happens only when every branch failed.

---

## 5. Cancellation

```
backend: cancel(run_id)
  -> RunStore.request_cancel(run_id)            sets cancel_requested on ITS side of the row
        (RunStore.save() NEVER writes that field, so the harness cannot erase it)
  -> harness sees it, two ways
        (a) step guard   at every step boundary, and immediately before EACH paid call
        (b) watcher      polls RunStore.cancel_requested every cancel_poll_ms (200) while a step runs
  -> watcher cancels the in-flight tasks         httpx aborts the provider request
  -> freeze the plan (nothing is removed)
        completed steps  -> "done"       keep outputs and spend
        in-flight step   -> "cancelled"  + branches_done, branches_aborted, partial_outputs
        pending steps    -> "skipped"
        cancelled_at_step set
  -> settle
        images with no Critic verdict -> orphaned "unchecked", NEVER returned
        images the Critic had allowed -> returned (best first)
        spend reported = actual spend only (the caller releases the rest)
  => Envelope CANCELLED { image_ids: [ checked images only ] }
```

Precedence: a block always wins. If any image was blocked the run ends REJECTED, even when a cancel
arrived later.

---

## 6. Gateway failover (Image Creator never learns a failover happened)

```
[Image Creator]   no LLM
  -> ModelGateway.generate(prompt, width, height, tier, seed, deadline_ms, chain_offset)
       -> chain = agents.yaml image_creator.models[tier] -> chains.yaml:
             1. cloudflare / flux-1-schnell        (free)
             2. cloudflare / flux-2-klein-4b       (free)
             3. openrouter / gemini-2.5-flash-image (PAID: skipped unless ALLOW_PAID=true)
          chain_offset skips leading entries (the switch_model lever)
       -> chain walk, for each entry:
             -> skip? provider exhausted today / model needs paid plan / paid and ALLOW_PAID off
             -> breaker OPEN? skip.   HALF-OPEN? exactly one probe call goes through
             -> call provider (httpx, shared deadline)
                   ok ........................ => image bytes, provider, model, neurons/cost
                   429 daily (3036) .......... mark Cloudflare exhausted until 00:00 UTC -> next
                   429 capacity (3040) ....... retry once with backoff -> next
                   5xx / timeout ............. retry once, trip breaker -> next
                   safety refusal (3030) ..... STOP the chain ----------> ContentRefused
                   HTTP 400 .................. STOP, our bug ------------> BadRequestError
       -> deadline runs out ................. -----------------------------> DeadlineExceeded
       -> every entry failed or skipped ..... -----------------------------> NoProviderAvailable
  -> Creator turns each exit into FAILED + output.failure:
       ContentRefused -> content_refused   NoProviderAvailable -> no_provider
       DeadlineExceeded -> deadline        BadRequestError -> bad_request
  -> the supervisor reads `failure` (section 3), never the exception text
```

Known: Cloudflare rejects a `seed` on flux-1-schnell (HTTP 400, code 5006), so that model is sent
none. FLUX.2 requests are always multipart/form-data.

---

## 7. Ports (the backend seam)

The harness imports only these Protocols. Each ships with a local implementation so everything
runs from a plain script.

```
Protocol         local implementation        the backend replaces it with           notes
---------------  --------------------------  -------------------------------------  ---------------------------------
RunStore         InMemoryRunStore            DB table keyed by run_id               TODO(backend): save() must never
  save(state)      keeps JSON per run_id       (harness writes everything except       write cancel_requested; the
  load(run_id)                                  cancel_requested)                       backend writes that column.
  cancel_requested(run_id)                                                            request_cancel() is its stand-in.

ArtifactStore    LocalArtifactStore          object storage + signed URLs           TODO(backend): url(image_id) is a
  put(bytes,mime) -> image_id                  (./.artifacts/ on disk)                 file:// URI locally; the backend
  get(image_id)                                                                        returns an https URL. Also owns
  url(image_id)                                                                        the orphan sweep.
  mark_orphaned(id, reason)                                                           reasons: unchecked, safety_block,
                                                                                       run_rejected, superseded, cancelled

EventSink        ListEventSink               SSE / queue transport                   TODO(backend): emit() is sync,
  emit(run_id, event)                          fan-out to GET /runs/:id/events         fire-and-forget; a sink that
                                                                                       raises can never fail a run.
```

Event shape (fixed): `{ run_id, step, public_label, phase, ts }`, `phase` is `started`, `progress`
or `finished`. `progress` is emitted per finished branch when N > 1. No internal agent name ever
appears in an event.

`create_images()` makes a **fresh in-memory run store and event sink per call** unless given
its own. A caller that wants to cancel a run or read its events must pass `run_store`, `events` and
a `run_id`.

---

## 8. Frontend <-> backend

The contract the frontend is built against. The backend (a teammate's repo) implements it by
calling `create_images()`. `scripts/dev_server.py` is a throwaway shim of it, dev only
(`DEV_FAKE=1` swaps the providers for a scripted fake so no `.env` or real call is needed;
run it with `uv sync --group server`, then
`DEV_FAKE=1 PYTHONPATH=src uvicorn --app-dir scripts dev_server:app --port 8000`).

```
browser (Next.js)
  -> POST /runs                      { prompt, plan, mode, n, settings }
       <= 202 { run_id }
  -> GET  /runs/:id/events           Server-Sent Events, replayed from the start
       <= event: step    data: { run_id, step, public_label, phase, ts }       (0..many)
       <= event: result  data: RunResult                                       (exactly once, last)
  -> POST /runs/:id/cancel
       <= 202 {}                      the backend calls RunStore.request_cancel(run_id)
  -> GET  /runs/:id
       <= RunResult                   (404 until the run exists; "running" while in progress)
  -> GET  /runs/:id/trace            developer inspector data, a SEPARATE channel (served only when
       <= Trace                       DEBUG_TRACE=1, else 404); live while the run is going
  -> GET  /images/:image_id
       <= the image file              (a URL for <img src>; JSON never carries bytes or base64)
          ^ create_images() runs here, in the backend, with the caller's RunStore/ArtifactStore/EventSink
```

```
CreateRequest { prompt: string,
                plan: "free"|"pro"|"max"|"ultra",
                mode: "auto"|"manual",
                n: 1..4,
                settings: { aspect_ratio?: "1:1"|"4:3"|"3:4"|"16:9"|"9:16", [question text]: answer } }

RunResult     { run_id, status, summary, reason|null,
                images:      [ { image_id, url, quality_score|null } ]   best first
                issues:      string[]            DONE_WITH_ISSUES
                questions:   string[]            NEEDS_INPUT
                assumptions: string[] }
```

A NEEDS_INPUT answer is sent by calling `POST /runs` again with the same request and each answer
added to `settings`, keyed by the question text.

### Envelope statuses and what the UI shows

| Status             | Meaning                                                                         | UI shows                                              |
|--------------------|---------------------------------------------------------------------------------|-------------------------------------------------------|
| `done`             | Every requested image was made, checked, and the best passes the quality bar.   | Image grid, best first, a score on each.              |
| `done_with_issues` | Images delivered, but the best is below the bar, some branches failed, or a retry could not finish. | Grid plus the `issues` list.            |
| `needs_input`      | The request needs something only the user can give (manual mode).               | A form for `questions`; submit re-runs with answers.  |
| `rejected`         | An image failed the safety review. Terminal, no images, never retried.          | A neutral "could not be delivered" message. No images.|
| `cancelled`        | The user (or backend) cancelled the run.                                        | Any images already checked; otherwise a short note.   |
| `failed`           | The run could not produce an image (provider down, budget, internal error).    | A short message and a Retry button.                   |
| `escalate`         | The router called it complex; the multi-step planner is not built yet.          | "This request needs more than one step, which is not available yet." |
| `running`          | Not an envelope status: the run is in progress (RunState only).                 | The three-step progress stepper and a Cancel button.  |

### Developer trace (inspector channel)

Not part of the product contract. It carries what the public channels must never carry (internal
agent names, the model that served each call, token details, decisions), so it is a separate
endpoint, off by default, and the frontend shows it only with `NEXT_PUBLIC_INSPECTOR=1`.

```
harness: Trace (one per run)  <- agents record each model call from data the gateway already returns
   - no extra model call; no image bytes (data: URLs become "[image omitted]")
   - writes into RunState.calls / .decisions / .orphaned, so it round-trips through JSON
   - router runs before RunState exists: buffered, then bound as calls[0] (seq 1)
shim:    GET /runs/:id/trace -> { calls, decisions, steps, attempts, budget, supervisor_spend,
                                   retries_used, critic_reasks_used, orphaned, status, totals, meta,
                                   created_at, finished_at, wall_ms }
frontend: Inspector polls it every 500 ms while the run is running, once more when it ends
```

One `CallRecord` per model call (LLM or image): `seq, ts, step, role, attempt, branch, chain,
provider, model, was_fallback, attempts[], prompt/completion/reasoning/total tokens,
tokens_estimated, reasoning_effort, neurons, cost_usd, duration_ms, was_repair, failure,
request_text, reply_text, reasoning_text` and, for images, `image { width/height requested and
returned, seed, seed_sent, seed_honoured, mime, bytes, steps }`. The supervisor's lever call is
`step: "supervise"`, kept apart from the step agents' calls.
