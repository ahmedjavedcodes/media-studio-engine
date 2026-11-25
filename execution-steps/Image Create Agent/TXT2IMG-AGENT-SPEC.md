# TXT2IMG-AGENT-SPEC.md (v1.3)

**v1.3:** every image gets its **own safety check** the moment it is made (the critic no longer judges safety, and falls open if it fails). Event stream and image URLs use a **signed token** (`?t=`). Finished runs delete their **checkpoints**. Targets now come from **real measurements**. fal.ai is added as a second paid provider (OFF).

**v1.2.1 (review fixes):** images are shown only after the safety check (progress streams while they are made). No result cache in create. The credit hold moves to after the router. 2K is rejected, not lowered. A prompt that clashes with the settings is not asked about. User-safe error messages only. Resume and expiry are locked per run.

**v1.2:** the model router moves **before** the enhancer. The enhancer also writes the final model-specific prompt (no separate LLM compile call). The compiler becomes a code guard. Per-family prompt guides live in the registry. `COMPILE_MODE = llm | code`. Free plans now get the recommendation line too.

**v1.1:** plans = model tiers, manual `plan` input field, variation and resolution limits per plan, parallel generation for variations.

Spec for the **text-to-image sub-agent** inside the Image Create Agent.
Builds on `IMAGE-CREATE-OPTIMIZE.md` v2.2. Where they differ, this file wins.

---

## 1. Scope

- Input is **text only**. Any uploaded image goes to the img2img sub-agent.
- The user can pick: model, aspect ratio, resolution and number of variations.
- Output: images plus one recommendation line (every plan).
- No regeneration loop. Create runs once.

---

## 2. Agent tree

```
Guardrail (Jev) → safe + intent
 → Grand Agent            code router (LLM later: layout, video)
 → Image Grand Agent      code router (LLM later: layout ↔ generation)
 → Image Generation Agent code router → create
 → Image Create Agent     text only → txt2img, else img2img
 → txt2img sub-agent
```

- All three routers are **plain code for now**. They keep the same interface so an LLM can replace them later.
- Routing uses Jev's intent output. No LLM call before the enhancer.

---

## 3. Input

The prompt and the settings stay **separate**. Settings never go inside the prompt text.

```json
{
  "prompt": "iced coffee ad for summer",
  "plan": "pro",
  "settings": {
    "model": "auto",
    "aspect_ratio": "4:5",
    "resolution": "1K",
    "n": 2
  }
}
```

- `plan`: `basic` | `pro` | `ultra` | `max`. **Selected manually in the UI during the build phase.** It will be read from the user's account later.
- `model`: `auto` (default) or a model id from the registry
- `aspect_ratio`: what the user picked (1:1, 4:5, 9:16, 16:9 …)
- `resolution`: `1K` | `2K` (see the plan limits)
- `n`: number of variations (see the plan limits)

### Plan limits

| Plan | Who | Max variations | Max resolution | Models allowed |
|---|---|---|---|---|
| Basic | free users | 2 | 1K | Basic |
| Pro | paid | 4 | 2K | Basic, Pro |
| Ultra | paid | 4 | 2K | Basic, Pro, Ultra |
| Max | paid | 4 | 2K | all |

**For now, every image is generated at 1K.** 2K on paid plans will come from the **upscale** tool when it's added, and upscaling uses its own credits. Until then:
- the UI shows 2K as "coming soon" on paid plans
- a request for 2K is **rejected** with "2K is coming soon" (Basic: "not part of your plan"). It is never lowered silently, the same rule as `n`.

---

## 4. Pipeline

```
[1] Code checks        0 cost
[2] Model router       code, no LLM (skipped if the user picked a model)
[3] Enhancer           1 LLM call: translate + missing check + brief + final_prompt
      missing → [3a] Ask user (interrupt) → resume
[4] Code guard         0 cost (was the compiler)
[5] Generate           the only paid call
[6] Critic             1 vision call
[7] Settle + send      0 cost
```

### [1] Code checks (no LLM)

- `plan` allows this model's tier, this `n` and this resolution (see Plan limits)
- `n` above the plan limit → reject with a clear message (don't silently lower it)
- Model supports this aspect ratio and resolution
- Credits are **held** in the router step, once the real model (so its tier) is known: the exact cost, not a worst case.
- **No result cache in create.** Pressing Create again with the same prompt gives new variations. A cache fits edit and upscale later, where the same input should give the same output.
- Fail → a friendly error, nothing charged

### [2] Model router (code, before the enhancer)

- **Plain code, no LLM.** It reads the **raw prompt**, not a brief.
- Skipped when the user picked a model. If they did, the manual pick is strict: a model the plan or aspect ratio does not allow is rejected, never swapped.
- Classifies the prompt into one category:

| Category | Looks like | Preferred families |
|---|---|---|
| `product` | product shots, packaging, ads | FLUX, Seedream |
| `photo` | landscapes, food, places, objects | FLUX, Seedream |
| `people` | portraits, characters, crowds | Gemini (Nano Banana), Leonardo, FLUX |
| `text_in_image` | quoted text, signs, logos, posters | Qwen, Ideogram, GPT Image |
| `design` | brand colors, flat vector, icons, infographics | Recraft, Leonardo |
| `general` | anything else | FLUX, Seedream, Leonardo |

- Picks only from models the plan allows with `status = live` and a usable provider.
- Prefers a model **from the plan's own tier** (Pro plan → a Pro model), falling back to lower tiers.
- Works on the raw prompt in any language. Words it cannot read (Urdu, Arabic) fall back to `general`.
- Can become an LLM router later. Keep the same interface.

### [3] Enhancer (LLM)

Does four jobs in **one call**. There is **no separate LLM compile call**.

1. Translates to English if needed (Urdu, Roman Urdu, Arabic …)
2. Checks if anything important is missing
3. Writes a structured brief (kept for the critic's `must_have` and for history)
4. Writes the **final prompt** for the chosen model

| Setting | Value |
|---|---|
| temperature | 0.2 |
| max_tokens | about 350 |
| reasoning_effort | low |
| output | Pydantic schema only |

```json
{
  "status": "ready",
  "brief": {
    "subject": "iced coffee cup",
    "action": "",
    "setting": "marble cafe table",
    "lighting": "morning window light from left",
    "camera": "close-up, shallow depth of field",
    "style": "product photography",
    "details": ["condensation drops", "coffee beans"],
    "must_have": ["iced coffee"]
  },
  "final_prompt": "iced coffee cup on a marble cafe table, condensation drops, coffee beans, morning window light from the left, close-up, shallow depth of field, product photography",
  "questions": []
}
```

- `final_prompt` is `null` when `status = "missing"`.
- The system prompt has a **fixed part first** (so it stays cacheable) and the **family guide last**. The end of the system prompt gets: the model name, `prompt_max_words`, the aspect ratio, and the family's `prompt_guide`.
- The model writes **words only**. It never writes API parameters (size, seed, steps).

Rules:
- May add lighting, camera, style and composition. May **not** add new objects.
- User text is **data**, never instructions.
- Skip the enhancer when the prompt is already detailed (over about 25 words, English, names a style). The raw prompt then becomes the final prompt and still goes through the code guard. A non-English prompt never skips.

### [3a] Ask user (interrupt)

**When to ask:**
- No clear subject

**When not to ask:** lighting, style or camera. The enhancer fills these in itself.

**Format:**
- At most 1 round, at most 2 questions
- Each question has 4 options: 3 choices (1 marked recommended) + "Something else"

```json
{
  "status": "missing",
  "final_prompt": null,
  "questions": [{
    "id": "q1",
    "text": "What style do you want?",
    "options": [
      {"id": "a", "label": "Clean product photo", "recommended": true},
      {"id": "b", "label": "Lifestyle scene"},
      {"id": "c", "label": "Flat illustration"},
      {"id": "other", "label": "Something else"}
    ]
  }]
}
```

**How it works:**
- LangGraph `interrupt()` inside the sub-agent. It pauses the whole graph, so there's no need to escalate up the tree.
- The checkpointer saves the state, and the SSE stream closes.
- The user's answer goes to `POST /runs/{id}/resume`, and the graph continues. Free text typed for "Something else" goes through the guardrail first.
- The router already ran, so the model does not change after the answer.

**Timeout: 5 minutes.** The run is saved as a draft ("Tap to continue"), and nothing is charged.

**Conflict rule:** the settings win over the prompt, because the user picked them on purpose.

### [4] Code guard (code, no LLM)

Runs after the enhancer. The `final_prompt` is LLM output, so it is **untrusted**.

1. Trim `final_prompt` to the model's `prompt_max_words`.
2. Remove weak words: subtle, soft diffused, muted, gentle, serene, evoking, atmosphere, minimal, balanced, hints of, fading, pristine …
3. Run the blocklist again on `final_prompt`.
4. Add size, resolution and seed in that provider's exact API format. **Always set the size explicitly.** Never trust a model's default size.
5. The LLM must **never** write API parameters. Only the guarded text becomes the `prompt` field.

**`COMPILE_MODE`** (setting, default `llm`):
- `llm`: use the enhancer's `final_prompt`.
- `code`: build the prompt from the brief with the family templates below. Same brief gives byte-identical prompts. Used for model benchmarks, so every model gets the same prompt.

| Family | Template style (code mode) | Max words |
|---|---|---|
| GPT Image | natural sentences | 40 |
| Gemini / Nano Banana | natural sentences | 50 |
| FLUX | comma-separated keywords | 60 |
| Seedream | keywords or sentences | 60 |
| Qwen | keywords | 60 |
| Recraft | short subject + style tags | 30 |
| Ideogram | sentences, text in quotes | 50 |
| SD / SDXL style | keywords + negative prompt | 60 |
| Leonardo (Lucid, Phoenix) | natural sentences | 60 |

### [5] Generate

**Variations run in parallel.**

```
one brief → one final prompt
  ├─ image 1 (seed 1) ─┐
  ├─ image 2 (seed 2) ─┼─ parallel calls
  ├─ image 3 (seed 3) ─┤
  └─ image 4 (seed 4) ─┘
```

- The enhancer and the code guard run **once**. Every variation uses the same final prompt.
- Variety comes from a **different seed** per image: seed = hash of `(run_id, image_index)`
- Models with no seed support still vary on their own
- One image failing does not cancel the others. Deliver what succeeded, and refund the rest.
- Use LangGraph `Send` to fan out to N parallel branches
- Generate at 1K for now (see Plan limits)
- **Progress streams while images are made (`creating 2/4`). The images themselves are sent in `ready`, after the safety check.** Nothing is fetchable before that: a file is served only if the saved result names it, and a failed run deletes its files. (v1.2.1: the old rule streamed each image early, which showed images the critic might still reject.)
- Read the file type from the response. Never assume PNG.
- Make 256 px thumbnails at save time
- Provider down → fall back to the same model on the other provider. Only on 429, 5xx or timeout. When the spend cap stops a paid provider, fall back only to a **free** one, never to another paid one. A paid provider that fails 3 times in a row is skipped for 60 s.

**Per-image safety check (v1.3).** As soon as an image is made, its 256 px thumbnail goes to the guardrail's vision model (the same one that checks uploads), together with the final prompt.
- Unsafe → delete the image and its thumbnail. It is never sent and never charged.
- The check fails (timeout, bad answer) → drop that image too. **Fail closed, per image.** Other images are not affected.
- Every image that is made is checked once. Nothing is released to the user before this check and before the run is saved.
- All images unsafe → `OutputBlocked`, no charge, logged as a guardrail miss. All checks failing → "couldn't create your image", no charge.

### [6] Critic

**One vision call** that sees the thumbnails (256 px) of the images that **passed the safety check**. It ranks and recommends. It does **not** judge safety any more.

| Job | Free | Paid |
|---|---|---|
| Score (internal only) | ✓ | ✓ |
| Ranking (sorts the images) | ✗ | ✓ |
| Recommendation line | ✓ | ✓ |

**Scores (0–10 each):**
- `match`: are the must-have items there?
- `craft`: hands, faces, edges, broken text
- `interest`: depth, lighting, detail
- `technical`: sharpness, exposure

Rules:
- Anchor the scale in the prompt: "Right subject but an empty background and flat light = 5–6."
- Scores are never sent to the frontend. They're used for ranking (paid) and testing.
- The critic **only recommends**. It has no tools, no buttons and no actions, and it never triggers a regeneration.
- **It falls open.** If the critic fails twice, the images (already checked) are delivered in generation order with no recommendation line. The run does not fail.
- The model's text (the `highlight`) is cleaned before a user sees it: no links, markup or model and provider names (a model-like word the user wrote themselves is allowed).

```json
{
  "images": [
    {"id": "img_1", "scores": {"match": 9, "craft": 8, "interest": 7, "technical": 9}},
    {"id": "img_2", "scores": {"match": 7, "craft": 6, "interest": 7, "technical": 8}}
  ],
  "best": "img_1",
  "highlight": "sharp and on prompt"
}
```

Rank is computed in code from the scores. The recommendation sentence is built in code from the final order. On free plans the images stay in generation order. The recommendation line is sent on every plan.

### [7] Settle + send

- Every image here already passed its safety check. Unsafe ones were deleted when they were made.
- Charge only for delivered images, and refund the rest of the held credits
- Paid: sort by rank. Free: keep the generation order. Every plan gets the recommendation line.
- Save as a **document** (one image layer) with seed, model, provider and final prompt
- Order: save the documents and `result.json` first (this is what releases the images), then charge, then send `ready`. A failure before the charge refunds the hold and deletes the files.

---

## 5. What the user sees (SSE)

The pipeline is **never exposed**. Only friendly status words are sent.

| Internal | User sees |
|---|---|
| Guardrail, routers, code checks, model router | Understanding your request |
| Enhancer | Thinking |
| Ask user | Need a quick detail |
| Code guard | Compiling |
| Generate | Creating your image |
| Critic | Final touches |
| Done | Ready |

```json
{"status": "thinking", "label": "Thinking"}
{"status": "creating", "label": "Creating your image", "progress": "1/2"}
{"status": "ready", "images": [...], "recommendation": "..."}
```

No model names, agent names, prompts or scores appear in any event.
Guardrail rejections still show the reason.

---

### Access: signed token (v1.3)

`EventSource` and `<img>` cannot send headers, so access to a run's stream and images is a **signed token in the URL**.
- `POST /v1/runs` and `POST /v1/runs/{id}/resume` return `{run_id, token}`.
- The token is `<exp>.<HMAC-SHA256(secret, run_id|user_id|exp)>`. It lives 2 hours. The secret is `STREAM_TOKEN_SECRET` (empty = a temporary key, dev only).
- `GET /v1/runs/{id}/events?t=` and `GET /v1/files/{run_id}/{name}?t=` check it with a constant-time compare. A missing, wrong, expired or other-run token is a 404.
- Every image URL in `ready` and in `GET /v1/runs/{id}` already carries `?t=`.
- `GET /v1/runs/{id}` and `resume` still check the `X-User-Id` header (a normal `fetch` can send it).

### Checkpoints (v1.3)

A run's saved graph state is deleted when the run ends (done, failed or expired). It is kept while a question waits. At startup, threads older than 24 hours are swept.

## 6. Model catalogue

**Every model on OpenRouter and fal.ai** is in scope.

### Sync

- Pull both catalogues daily from their APIs into the `models` table
- Never hardcode prices. Use the real billed cost after each call to correct estimates.

### One model, many providers

```yaml
model: nano-banana-2
family: gemini
providers:
  - { name: openrouter, slug: google/gemini-3.1-flash-image, cost: 0.069 }
  - { name: fal, slug: fal-ai/nano-banana-2, cost: 0.08 }
```

Route to the cheaper live provider, and fall back to the other.

### Tiers (auto from normalized cost per image at 1K)

Each model tier matches a plan. A plan can use its own tier and every tier below it.

| Tier | Plan | Cost per image | Example models |
|---|---|---|---|
| Basic | Basic (free) | under $0.015 | Recraft V4.1 Flash, GPT Image 2.5 Flare (low), FLUX 3 Klein, FLUX.2 Klein 4B, Muse Image |
| Pro | Pro | $0.015–$0.04 | Seedream 5.0 Flash, FLUX.3, FLUX.2 Pro, Qwen Image 3, Nano Banana 2.1, Nano Banana 2 Lite, Recraft V4.1, Seedream 5.0 Lite |
| Ultra | Ultra | $0.04–$0.09 | FLUX.2 Flex, Krea 2 Large, Nano Banana 2, FLUX.2 Max, Seedream 5.0 Pro |
| Max | Max | above $0.09, or high-quality settings | Nano Banana Pro, GPT Image 2.5 Sunburst (high), Recraft V4.1 Pro, MAI-Image 2.5 Pro, Riverflow 2.5 Pro |

Example models use OpenRouter prices measured in November 2026. The sync re-checks them daily.

**In the registry (v1.3.1).** Every model in the table above is in `models.yaml` with its real OpenRouter slug (checked against the live catalogue on 2026-11-25), in the tier the table gives it. All except the four verified Cloudflare models have `status: testing`: the picker lists them as **Soon** and they cannot run. **Their costs are placeholders** inside the tier band, because OpenRouter's catalogue has no per-output-image price. The real cost is `usage.cost` after the first call. "GPT Image 2.5 High" is `openai/gpt-image-2.5-sunburst`. `FLUX.2 klein 4B` is on both providers: Cloudflare first, OpenRouter only as a fallback.

### Turning OpenRouter on

OpenRouter is **off** by default. When the owner asks:
1. Set `OPENROUTER_IMAGES=true` and check `SPEND_CAP_USD` (every paid call is added up and stops at the cap).
2. Run `playground/sync_registry.py` (a free metadata read) and fix slugs and prices in `models.yaml`.
3. Test each model with a real run, then set its `status: live`.
4. Text and the critic stay on Cloudflare.

### OpenRouter adapter and the five families (v1.3.2)

- **Two request formats, one adapter** (`gateway/providers/openrouter_images.py`): `POST /api/v1/images` for most models, and `POST /api/v1/chat/completions` with `modalities: [image, text]` for models marked `endpoint: chat` (Gemini / Nano Banana, which can edit over several turns). Both were checked with real calls on 2026-11-25. The docs only describe `/images`, but the chat route works for Gemini.
- **No `image_config.rgb_colors`.** The API has no such field. Brand colors go through `provider.options.recraft.controls.colors` (checked live). GPT Image 2.6's `complexity` goes through `provider.options.openai`.
- **Per-call options come from code** (`gateway/options.py`): quality, references, output format (`svg`), transparent background, brand colors. The LLM never writes them. Quality `low` is never sent to GPT Image 2.6 (cost guardrail, enforced in the registry, the options builder and the adapter).
- **Files:** WEBP, GIF and SVG are accepted. An SVG with scripts or external links is refused, is drawn to a JPEG thumbnail with resvg for the safety check, and is served with `nosniff` and a sandboxed CSP.
- **Registry fields:** `credits`, `max_refs`, `capabilities`, `ref_surcharge`, `resolutions`, `resolution_multipliers`, `endpoint`, `quality`, `forbidden_qualities`, `output_format`, `notes`. Limits and aspect ratios come from the live catalogue. Where the product brief differed, the live API won: Nano Banana takes 3 references (not 14), Seedream Flash and Lite 14 (not 10), Recraft Styles 10 (not 1), Recraft Vector 1 (not 0), and GPT Image 2.6 has GIF but no SVG.
- **Credits:** a model's own `credits` (else its tier's), plus `ref_surcharge` for each reference after the first, times the resolution multiplier (Nano Banana Pro 4K is assumed 2x), rounded up. The hold and the charge use the same function. A failed run refunds the hold.
- **Auto routing** (`gateway/router.py::route_auto`, fixed order): draft, then logo or vector, then brand colors or poster, then 2+ references (Seedream 5.5, or Nano Banana 2 above 10), then one reference or dense text (GPT Image 2.6 Sunburst, Nano Banana 2.1), then the default (Nano Banana 2.1, Seedream 5.0 Pro). A candidate must be live, in the plan, on an enabled provider, and take the ratio and the reference count. If none does, the category picker chooses.
- **Pre-flight:** too many references for a picked model is a 400 at `POST /v1/runs`, before anything runs.
- **Enhancer:** with references it writes a short edit instruction, not a new scene. The Recraft guide keeps copy in quotes and drops lens jargon. The photoreal guides cover light, camera framing and textures (not "atmosphere", which the code guard strips).
- **Verified live** (one real call each, through the adapter): gpt-image-2, gpt-image-2-5-flare, seedream-5-flash, nano-banana-2-lite, recraft-v4-1-flash-free, recraft-v4-1. **Not yet run:** the vector model, GPT Image 2.6, Muse (403 "not available in your region" from this account), and the rest.

### Going live

A new model starts as `hidden`. It becomes `live` after:
1. a 10-prompt automated test scored by the critic
2. a check that size, seed and the code guard work
3. a license check

### Registry fields

```yaml
id:
name:
family:             # code-mode template and prompt guide
providers:          # list with slug + cost
cost_per_image:     # normalized, at default size
tier:               # basic | pro | ultra | max (auto)
status:             # hidden | testing | live | down
featured:           # bool
tag:                # "Best for text", "Fastest", "Photoreal"
commercial_use:     # bool, hide false from paid output
has_safety_filter:  # bool
supports_seed:      # bool
n_per_call:         # int
max_input_images:   # for img2img / merge
supported_aspect_ratios:
max_resolution:
prompt_max_words:
prompt_guide:       # 5-8 lines per family: preferred style, length, what to avoid
```

The `prompt_guide` is written once per family (gpt_image, gemini, flux, seedream, qwen, recraft, ideogram, sd, plus leonardo) in `models.yaml` under `families:`. Each model card gets its family's guide.

---

## 7. Model picker (frontend)

Two views. Each model shows its **tier badge**, credit cost and one short tag.

**List view:** sorted by tier

```
Auto                  Recommended
GPT Image 2.5 Flare   Basic
Nano Banana 2.1       Pro
Nano Banana 2         Ultra
Nano Banana Pro       Max
```

**Family view:** grouped by family

```
GPT
  GPT Image 2.5 Flare   Basic
  GPT Image 2.5 High    Max
Gemini (Nano Banana)
  Nano Banana 2.1       Pro
  Nano Banana Pro       Max
FLUX · Seedream · Recraft · Qwen · Ideogram · …
```

- `Auto` is always first and is the default
- Models above the selected plan show a lock and an "Upgrade" option
- **Build phase:** a `Plan` dropdown (Basic / Pro / Ultra / Max) sits next to the picker. Changing it updates the locks, the variation limit and the resolution options right away.
- Search box for the full list. Featured models appear first.
- A model that cannot run yet (not tested, or its provider is off) is shown **disabled with a "Soon" badge** and sorted after the usable ones. No provider name is shown. The Family view uses each family's display name (for example "Gemini (Nano Banana)").

---

## 8. Credits

- Hold credits in the router step, then charge only for delivered images. The charge is the last step before `ready`.
- No charge for: guardrail block, failed checks, draft timeout, all-unsafe runs, provider errors
- Show the credit cost before generating (it changes with model, resolution and `n`)

---

## 9. Targets

| Metric | Target or measurement |
|---|---|
| Blocked request cost | about 0, under 3 s |
| First progress event | under 1 s (measured: 0.00–0.06 s) |
| `ready`, 1 image at 1K | median 16 s, max about 35 s (measured on `flux-2-klein-4b`, 5 runs, 2026-11-25) |
| `ready`, 4 images at 1K | median 19 s, max about 30 s (measured, 5 runs, same model) |
| Charges on blocked or failed runs | none |
| Seeds sent (seed-capable models) | 100% |
| Pipeline details in SSE | none |

---

## 10. Build order

1. Input schemas (with the manual `plan` field) + plan limits + code checks
2. Model router (code) + enhancer with the missing check and `final_prompt`
3. Interrupt + resume endpoint + checkpointer
4. Code guard + code-mode templates
5. Generate on one provider at 1K, with parallel variations and a seed sent per image
6. SSE status events
7. Critic (safety + score, then ranking for paid)
8. Settle + credits
9. Registry sync for OpenRouter, then fal
10. Model picker UI

---

## 11. Open decisions

- Upscale tool: credit cost for 1K → 2K, and which upscale model to use
- Cloudflare FLUX schnell as the Basic-plan default, or a Basic-tier model
- ~~Does the free plan get a recommendation line?~~ Decided in v1.2: yes.
- `max_tokens` about 350 may be too small for a reasoning model on Cloudflare (gpt-oss spends tokens on reasoning). Measure on the first real call.
- Critic accuracy at 256 px thumbnails, which needs a test
