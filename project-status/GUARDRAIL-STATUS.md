# Guardrail Agent status

Spec: `execution-steps/GUARDRAIL-AGENT-SPEC.md` (section 10 replaced by the strike rules below).
Branch: `feat/guardrail-agent`. Last updated 2026-11-24.

## Phase 1 checklist

- [x] **Input guardrails (text + image)**: built, wired into `POST /v1/runs`, batch-tested on the live API
- [ ] Prompt Enhancer / Create Agent / generation (later steps, not started)

## Spec "Done when" (section 15)

- [x] No path reaches the generator without APPROVE from every check (`guard()` raises on anything else; unit + API tests)
- [x] Generator gets the raw prompt and original images, byte for byte (unit + API tests)
- [x] Any provider failure gives 503, 0 credits, no strike (unit tests, and seen live). No credits system exists yet, so "0 credits" is by construction
- [x] Agent never exceeds 2050 ms (unit test with both checks hanging; live max 1724 ms)
- [x] Every 422 has `category`, `blocked_by`, `message` and `tip` (`tip` is null for CSAM and hate by design)
- [x] Raw `reason` and `scores` never in API responses (tested)
- [ ] 0 misses on CSAM and NCII red-team prompts: text is 0 misses (T06, T07 CSAM). **Image NCII cases I03-I06 were not run** (fixture `person_adult_synthetic.jpg` missing)
- [x] `ruff`, `mypy`, `pytest` pass (186 tests)
- [ ] Works in the frontend: screen builds and lints; **not yet clicked through in a browser by a person**

## Strike and cooldown rules (replace spec section 10)

- Instant reject: 1 strike. Strikes 1-4 warn ("Strike N of 5"). Strike 5 in 30 days bans.
- Standard reject: no strike. 60 s cooldown, 120 s if another standard reject in the last 10 min.
- 5 rejects of any kind in 10 min: 30 min cooldown. A new cooldown never shortens a running one.
- 503s and requests blocked by a cooldown are not counted as rejects.

## Batch test results (2026-11-24)

Details in `test-manual/guardrails/result/`.

| Group | Result |
|---|---|
| Text must-block (18) | 18/18 exact decision and category |
| Text must-allow (11) | 11/11 approved |
| Images | 4 answered as expected; 2 returned 503 (vision timeout); I09 unknown (503) |
| Bad input, image prep, strikes, cooldowns | all pass |

Latency: text p50 420 ms, max 505 ms. Images p50 about 1.5 s. Cost: Cloudflare $0 (free tier); Jev 43 calls, 55,741 in / 12,943 out tokens (TypeSafe price unknown).

## Open items

1. **Timeouts.** The spec's 500 ms Jev timeout fails from a cold connection (about 870 ms) and sometimes warm (max 505 ms). The batch ran with `JEV_TIMEOUT_S=1.5`. Suggest about 1000 ms plus opening the connection at startup.
2. **Vision timeout.** Scout takes about 1.4-1.5 s against a 1.5 s limit, so most image checks fail in build mode. Real Llama Guard 4 (OpenRouter, paid) is untested.
3. **Jev in build mode.** Cloudflare has no Jev model. Using it needs `JEV_PROVIDER=typesafe` (TypeSafe is not a free provider).
4. **Missing fixture** `person_adult_synthetic.jpg`, so I02-I06 are untested. I03-I06 are the image-undressing must-block cases.
5. **Unspecified CSAM combo rule** in `policy.py`: score = min(involves_minor, sexual/undress). Needs owner sign-off.
6. Unsafe Llama Guard answers below p=0.60 are approved (spec section 8 table). Consider a floor for CRITICAL categories.
7. Strike ledger is in memory (resets on restart). Real persistence and token revocation are not built.
8. "Report a wrong block" link has no backend (`NEXT_PUBLIC_REPORT_URL`).
9. Evals `evals/run_guardrail_eval.py` and the test-manual dataset are separate sets; merge or pick one.
