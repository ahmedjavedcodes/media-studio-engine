# Spec: Guardrail Agent (Text + Image Input)

Phase 1 · Read with `CLAUDE.md` and `IMAGE-INPUT-GUARDRAILS.md`

## 1. Goal
Check every request before any generation call.
- Text only → **Jev** (TypeSafe).
- Images + text → **Jev** on text and **Llama Guard 4 12B** on images + cleaned text, in parallel.
- Strictest answer wins. Blocked or failed checks cost 0 credits.
- If approved, the generator gets the **raw prompt** and **original images**.

## 2. Facts that change the plan
- Llama Guard 4 returns text (`safe`, or `unsafe` + codes like `S1,S12`), not JSON. Parse it.
- Groq retired Llama Guard 4 (2026-03-05). Use OpenRouter `meta-llama/llama-guard-4-12b`.
- Llama Guard 4 does not detect injection, hidden image text, faces or watermarks. Those stay in other layers.
- Jev image input is beta. Send Jev text only.
- Cloudflare has no Llama Guard 4. Build mode uses a stand-in (§5).

## 3. Flow
```
0. Account check      frozen/banned → 403 · cooldown → 429
1. Clean text         keep prompt_raw, build prompt_clean
1b. Prep images       validate, strip EXIF, downscale to 768
2. Run together       Jev(text) + LlamaGuard4(images + prompt_clean), deadline per section 9
3. Merge              strictest wins
4. Act                APPROVE → generator · reject → 422 + strike rules
```

## 4. Files
```
src/app/guardrails/
  agent.py         check(GuardInput) -> GuardDecision
  clean.py         text cleaner (pure)
  image_prep.py    image prep (pure)
  text_check.py    Jev → GuardVerdict
  vision_check.py  Llama Guard 4 → GuardVerdict
  policy.py        tiers, thresholds, S-code map
  merge.py         strictest wins
  strikes.py       strikes and cooldowns
  schemas.py
src/app/gateway/providers/typesafe.py, openrouter.py, cloudflare.py
src/app/contracts/ports.py        + StrikeLedger
src/app/local/fake_strikes.py
prompts/guardrail/jev_questions.yaml
prompts/guardrail/vision_standin.md
prompts/guardrail/user_messages.yaml
```
All model calls go through `gateway/` with `httpx`.

## 5. Providers
| Check | Build mode | Test mode |
|---|---|---|
| Jev | **TypeSafe direct** (decided 2026-11-24): `POST https://api.typesafe.ai/v1/systemone`, model `jev-1.13.0`. Cost is about $0.00005 per check | Same |
| Vision | Stand-in: `@cf/meta/llama-4-scout-17b-16e-instruct` told to answer in Llama Guard format. Not a safety guarantee | OpenRouter `meta-llama/llama-guard-4-12b` |

Pin versions. No fallback model. Failure = fail closed.

Build mode calls Jev directly through TypeSafe. The Cloudflare route `typesafe/jev` on `/ai/run` returned "No route for that URI", so it is not used (`JEV_PROVIDER=cloudflare` stays as an option but is unverified).
Each provider has one shared `httpx.AsyncClient` with keep-alive, and API startup sends one warm-up call to Jev and to the vision model. A cold TLS connection costs about 870 ms.

## 6. Schemas
```python
Decision = APPROVE | STANDARD_REJECT | INSTANT_REJECT
Category = SAFE | CSAM | NCII_UNDRESSING | GORE_SELF_HARM | TERRORISM_WEAPONS
         | NSFW_EXPLICIT | CELEBRITY_LIKENESS | HATE_HARASSMENT
         | PROMPT_INJECTION | PII_WATERMARK | OTHER_UNSAFE
Tier     = CRITICAL | SEVERE | STANDARD | NONE

class GuardVerdict(BaseModel):          # one per model call
    source: Literal["jev", "vision"]
    decision: Decision
    category: Category
    confidence: float | None            # None = model gave no score
    reason: str                         # internal log only
    scores: dict[str, float]
    needs_review: bool = False
    model: str
    latency_ms: int

class GuardDecision(BaseModel):         # final
    decision: Decision
    category: Category
    confidence: float | None
    verdicts: list[GuardVerdict]
    account_action: Literal["none", "warn", "cooldown_30m", "cooldown_24h", "ban"]
    user_message: str | None
```
The user always sees why they were blocked (§11). They see the fixed message for the category, not the model's raw `reason` or `scores`. Those stay in the audit log.

## 7. Steps

### 7.1 Clean text
1. Over 4000 chars or bad UTF-8 → 400.
2. NFKC. Remove zero-width and control characters. Lowercase. Collapse spaces.
3. Join spaced single characters: `n u d e` → `nude`.
4. Undo leetspeak (`0→o 1→i 3→e 4→a 5→s 7→t @→a $→s`) only on tokens mixing letters and digits/symbols. Skip `4k 8k 2d 3d 1080p 16:9 9:16`.
5. No translation. Target < 5 ms.

### 7.2 Jev
- `state = {prompt_raw, prompt_clean, has_image}`.
- One `noul` question per category, worded so **yes = violation**: `csam`, `ncii_undressing`, `gore_self_harm`, `terrorism_weapons`, `nsfw_explicit`, `celebrity_likeness`, `hate_harassment`, `prompt_injection`, `pii_watermark`.
- Put ALLOW examples in `criteria.false` (fantasy weapons, swimwear, fashion, statues, own portrait, synthetic people).
- Read `answers[key].noul` (0 to 1). Missing or invalid → fail closed.

### 7.3 Image prep
1. JPEG, PNG or WEBP by magic bytes. Max 20 MB, 50 MP. Else 400.
2. Apply EXIF rotation, strip metadata, convert to RGB (alpha on white).
3. Longest side > 768 → resize to 768 (Lanczos). Never upscale.
4. JPEG q90, base64 data URL. Max 5 images per call; more → parallel batches.

### 7.4 Llama Guard 4
- `temperature=0`, `max_tokens=10`, `logprobs=true`. One user message: text + image(s).
- Parse: `safe` → APPROVE. `unsafe` → map codes below. Anything else → fail closed.
- Confidence = probability of `unsafe` from logprobs. No logprobs → `None` → STANDARD_REJECT + `needs_review`, no strike.

| Codes | Category | Tier |
|---|---|---|
| S4 | CSAM | CRITICAL |
| S9 | TERRORISM_WEAPONS | CRITICAL |
| S3 | NCII_UNDRESSING | SEVERE |
| S11 | GORE_SELF_HARM | SEVERE |
| S12 | NSFW_EXPLICIT | STANDARD |
| S10 | HATE_HARASSMENT | STANDARD |
| S5 | CELEBRITY_LIKENESS | STANDARD |
| S7, S8 | PII_WATERMARK | STANDARD |
| S1, S2, S6, S13, S14 | OTHER_UNSAFE | STANDARD |

## 8. Decision rules
| Tier | p ≥ 0.85 | 0.60 ≤ p < 0.85 | p < 0.60 |
|---|---|---|---|
| CRITICAL, SEVERE | INSTANT_REJECT | STANDARD_REJECT | no action |
| STANDARD | STANDARD_REJECT | STANDARD_REJECT | no action |

Merge:
1. `INSTANT > STANDARD > APPROVE`. Highest wins. Tie → higher tier, then higher confidence.
2. APPROVE only if every check returned APPROVE.
3. Any INSTANT → cancel the other task, return.
4. A reject from one check beats a failure in the other. No reject + any failure → 503.

## 9. Timeouts
Per `PROVIDER_MODE` (values from the 2026-11-24 eval review; env overrides in `core/config.py`):

| Part | Build (Scout stand-in) | Test (Llama Guard 4) |
|---|---|---|
| Jev | 800 ms, retry after 400 ms | same |
| Vision | 2500 ms, no retry | 1500 ms, retry after 800 ms (placeholder, measure first) |
| Whole agent | text only 1500 ms; with images 3000 ms | 2000 ms (placeholder, set after measuring) |

- Attempt timeout = `min(timeout, deadline − now)`. Skip retry if < 100 ms left.
- Retry on timeout, 429, 5xx. Not on 400.
- Fail closed → 503 `"Safety verification timed out. Please try again."`, 0 credits, no strike. Never fail open.

## 10. Strikes and cooldowns (via `StrikeLedger` port, idempotent by `request_id`)
Replaced 2026-11-24. Tiers still decide INSTANT vs STANDARD (§8). They do not change these limits.

| Trigger | Action |
|---|---|
| INSTANT reject (any category) | 1 strike. Count strikes in the last 30 days. Strikes 1 to 4: block + warning "Strike N of 5. At 5 strikes your account will be banned." Strike 5: freeze, revoke tokens, permanent ban |
| STANDARD reject | No strike. 1 minute cooldown on the first one; 2 minutes if there was another standard reject in the last 10 minutes |
| 5 rejects of any kind in 10 min | 30 minute cooldown |

A new cooldown never shortens a running one. 503s and requests blocked by a cooldown do not count as rejects.
Settings: `INSTANT_STRIKE_LIMIT=5`, `STRIKE_WINDOW_DAYS=30`, `STANDARD_COOLDOWN_SEC=60`, `STANDARD_REPEAT_COOLDOWN_SEC=120`.

## 11. Errors
| Case | HTTP | `code` |
|---|---|---|
| Bad input | 400 | `bad_input` |
| Frozen or banned | 403 | `account_restricted` |
| Cooldown | 429 | `cooldown` + `retry_after` |
| Any reject | 422 | `content_blocked` (+ `warning` on strike 1) |
| Guardrail failed | 503 | `guardrail_unavailable` |

### 11.1 Reject reason shown to the user
Every 422 body tells the user why:
```json
{
  "code": "content_blocked",
  "category": "NSFW_EXPLICIT",
  "blocked_by": "prompt",
  "message": "Your request asks for nudity or sexual content.",
  "tip": "Try a fully clothed version.",
  "warning": false
}
```
- `blocked_by`: `prompt` when Jev rejected, `image` when Llama Guard rejected, `both` when both did.
- `message` and `tip` come from the table below, stored in `prompts/guardrail/user_messages.yaml`. Never send the model's raw `reason`, scores or thresholds.
- Instant and standard rejects use the same message. On strike 1, set `warning: true` and add: "Repeated requests like this will lead to an account ban."

| Category | `message` | `tip` |
|---|---|---|
| CSAM | Your request may involve a minor in sexual or suggestive content. | none |
| NCII_UNDRESSING | Your request asks to remove or change a real person's clothing, or put a face on an explicit body. | Edit the background, style or lighting instead. |
| GORE_SELF_HARM | Your request contains graphic gore, self-harm or cruelty. | Try an action scene without gore. |
| TERRORISM_WEAPONS | Your request relates to making weapons or terror attacks. | Sci-fi and fantasy weapons are fine. |
| NSFW_EXPLICIT | Your request asks for nudity or sexual content. | Try a fully clothed version. |
| CELEBRITY_LIKENESS | Your request shows a real public figure. | Try a fictional or AI-made person. |
| HATE_HARASSMENT | Your request contains hateful or harassing content. | none |
| PROMPT_INJECTION | Your request tries to change the system's rules. | Describe only the image you want. |
| PII_WATERMARK | Your request contains an ID card, bank card or watermarked image. | Use your own photo or a plain mockup. |
| OTHER_UNSAFE | Your request may break our content policy. | Try rewording your prompt. |

The frontend shows `message`, `tip` and the warning banner, plus a "Report a wrong block" link.

## 12. Out of scope
OCR on image text, face/celebrity/child detection, logo and watermark detection, Prompt Guard 2, word lists, combo rules. See `IMAGE-INPUT-GUARDRAILS.md`.

## 13. Logging
LangSmith trace per check (models, decision, category, latency per step). Audit row per reject with all verdicts and image `sha256`. Never store rejected image bytes or API keys.

## 14. Build order (pytest + respx for each)
1. `schemas`, `policy`, config, errors
2. `clean`
3. `image_prep`
4. Jev provider + `text_check`
5. Vision providers + `vision_check`
6. `merge`
7. `StrikeLedger`, fake, `strikes`
8. `agent` (parallel, deadline, retries, early exit)
9. Wire into `POST /v1/runs` and the frontend
10. Evals from `IMAGE-CREATE-TEST-PROMPTS.md` + false-positive set + Urdu/Roman Urdu prompts

## 15. Done when
- [ ] No path reaches the generator without APPROVE from every check
- [ ] Generator gets raw prompt and original images, byte for byte
- [ ] Any provider failure → 503, 0 credits, no strike
- [ ] Agent never exceeds its deadline plus 50 ms (§9: text 1500 ms, images 3000 ms in build mode)
- [ ] Every 422 has `category`, `blocked_by`, `message` and `tip` from the table
- [ ] Raw `reason` and `scores` never in API responses
- [ ] 0 misses on CSAM and NCII red-team prompts
- [ ] `ruff`, `mypy`, `pytest` pass; works in the frontend

## 16. Open decisions
1. ~~Build-mode Jev: Cloudflare `typesafe/jev` or TypeSafe direct?~~ **Settled 2026-11-24: TypeSafe direct** (see §5).
2. Build-mode vision: Scout stand-in or OpenRouter `llama-guard-4-12b:free`?
3. CRITICAL bans: instant final, or freeze then human review?
