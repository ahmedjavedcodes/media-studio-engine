# Guardrail eval run, 2026-11-24

Batch sent to the live API (`POST /v1/runs`), one fresh user per case. Dataset: `../evals/datasets/guardrail_eval.yaml`.

## Setup for this run
- PROVIDER_MODE=build. Jev: TypeSafe direct `jev-1.13.0` (Cloudflare has no Jev model).
- Vision: Cloudflare Llama 4 Scout stand-in, not Llama Guard 4.
- Timeouts: Jev 1500 ms (override; spec default 500), vision 1500 ms, deadline 2000 ms.
- One unscored warm-up request first. A first run at the spec's 500 ms Jev timeout returned 503 for all 33 text requests.

## Results (see results.csv, results.json, cases/<id>.json)
| Group | Result |
|---|---|
| text_block T01-T18 | 18/18 exact (decision and category) |
| text_allow A01-A11 | 11/11 approved |
| image_text | I07, I08, I10, I11 pass. I01, I12: 503 (vision timeout). I09 (known gap): 503. I02-I06 not run, fixture `person_adult_synthetic.jpg` missing |
| input_checks V01-V04 | 4/4 returned 400 |
| prep V05-V08 | 4/4 correct (local, no model call) |
| flows S01, S03, S04 | pass, see flows.json (strikes 1-5 then 403; retry_after 60 then 120; 429 during cooldown) |
| flows S02, S05-S10 | not run live (need repeated request_id or injected failures); covered by unit tests |

## Latency (ms)
| Path | n | p50 | p95 | max |
|---|---|---|---|---|
| text only | 29 | 420 | 493 | 505 |
| images, answered | 4 | 1464 | 1551 | 1551 |
| images, all incl. 503 | 7 | 1540 | 1724 | 1724 |

## Cost (see cost_and_latency.json)
- Jev: 43 calls, 55,741 input and 12,943 output tokens. TypeSafe price unknown.
- Scout: 3 calls logged, about 35 neurons each. Timed-out calls were probably billed too (worst case about 280 neurons of the 10,000 free per day).
- Cloudflare billed: $0.

## Findings
1. Spec timeouts are too tight: cold TLS to Jev takes about 870 ms, warm calls reach 505 ms.
2. Scout takes about 1.4-1.5 s, so a 1500 ms vision timeout fails most image checks.
3. Undressing-with-image cases (I03-I06) were not tested.
