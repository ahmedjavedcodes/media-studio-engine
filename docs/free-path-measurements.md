# Free-path measurements against IMAGE-CREATE-OPTIMIZE §12

Live runs on 2026-11-20: real Groq + Cloudflare calls, plan `free`, `ALLOW_PAID` off, no money
spent (paid ledger untouched: $0.034 of $5.00). Raw data: `.cache/measure/*.json` (not committed).
Scripts: `playground/measure_free_path.py`, `e4_position_bias.py`, `e1_thumbnail_size.py`.

Sample sizes are small (5 to 6 runs per row). Read them as measurements, not statistics.

## §12 table

| metric | baseline | target | measured | met? |
|---|---|---|---|---|
| cost to refuse an unsafe request | 549 neurons / 42 s | ≤ 10 neurons / ≤ 3 s | **0 neurons / 1.9 s** (1 call, 547 tokens, not billable) | yes |
| LLM calls, first-pass run | 6 | 3 | **3** at n=1 and n=2; **4** at n=4 (the vision model takes ≤ 3 images a request, so 4 images are two Critic calls); +1 per repair | n≤2 yes |
| wall time, 4 images | 41.8 s | ≤ 17 s | **49 to 55 s** with the Critic falling back to Cloudflare (see below); about 14 to 18 s estimated when Groq answers both Critic calls | **no** |
| time to first image | 41.8 s | ≤ 8 s | **8.7 to 11.4 s at n=1** (gate 1.0–1.5 s, brief 1.2–1.7 s, image 4–7 s, critic 1.3–1.7 s); equals the wall time at every n | **no** |
| free images/day | 73 | ≥ 170 | **about 130 at best.** Cloudflare allows 173 images (57.6 neurons each of 10,000) but the Critic's Groq model allows 200,000 tokens/day at about 1,470 per image = 136. Today's runs used it up | **no** |
| critic score spread, mixed set | 0 | ≥ 2 | **3 to 10 across the set** (scores seen 3, 4, 6, 7, 7, 8, 9, 9, 10). Inside one run, spread ≥ 2 in 2 of 6 multi-image runs | yes on a mixed set; no inside one run |
| seeds honoured | 0% | 100% | **0%.** No model reproduces from a seed (Cloudflare schnell rejects it, Cloudflare Klein and OpenRouter Klein ignore it) | **no, not achievable** |
| blocked run charges credits | yes | no | `billable: false`, 0 neurons | yes |
| compiled prompt within budget | 70 words | within each model's | enforced by the Compiler and tested for every style and budget | yes |

## What the live runs found (all now fixed, each with a test)

1. **Groq qwen takes at most 3 images per request.** A 4-image request is an HTTP 400, so n=4
   failed at the Critic. The Critic now reviews balanced groups of at most 3 (`max_images_per_call`).
2. **qwen needs to see the schema.** With `json_schema` only, it answered with invented field names.
   It now gets the schema both ways (5/5 valid afterwards).
3. **`json_validate_failed`.** Groq's schema mode returns HTTP 400 when the model emits broken JSON
   (`{~~ ...`). We treated that as our own bad request. It is now an ordinary bad reply that the
   repair path handles. Seen 1 time in about 12 Critic calls.
4. **The free budget refused half of n=4**: the critique estimate (12,000 tokens/image) was 2.4x
   the measured worst case. Measured: about 560 fixed + 1,280 per image prompt tokens, 70 + 25 per
   image in the reply, a repair about doubles it. Set to 5,000 tokens and 50 neurons per image,
   and the Critic's own step budget is that times the image count. Free n=4 now fits.
5. Unreadable "block" replies (fail the whole batch): **0 occurrences** in about 17 Critic calls.

## Still open, from the data

- **n=4 is slow because of a rate limit, not the models.** Two Critic calls of about 3,100 input
  tokens each reach Groq's 7,000-input-tokens-a-minute limit for qwen, so the second spills to
  Cloudflare llama-4-scout: 39 s and about 60 neurons. This also means about one n=4 run a minute
  fits in the whole app. Options: serve the Critic from a model with a higher limit, ask for the
  paid Groq tier, or accept the Cloudflare fallback and its neurons.
- **Time to first image cannot reach 8 s with a ranking Critic.** An image is shown only after the
  Critic has seen it, and the Critic waits for the last image of its group. Image generation alone
  is 4 to 7 s on Cloudflare. Possible: review group 1 as soon as its images exist (stream per group).
- **Critic scores are noisy** (E1 below): the same three images scored 7/3/3, 3/6/6 and 6/4/4 on
  three consecutive calls. Do not set tier thresholds from single readings: E5 needs repeats.

## Checks of the spec's build order (§13)

- **Item 4, document output (§4.3): done.** `output.documents[]`: one image layer each, ranked,
  `scores` carries the four dimensions plus `quality_score`. Tested in `test_supervisor.py`,
  `test_linear.py`.
- **Item 6, stream each image as it finishes: partly.** `image_ready` events exist, carry the image
  id, and fire only for images the Critic allowed. But the Critic now reviews a whole group in one
  call, so every `image_ready` of a group fires together at the end. Measured first image = wall time.
- **Item 10, one guard per step (D5): done.** `_gate` decides cancel and budget once per step,
  takes the branch count, logs once; `_branch_gate` re-checks only cancel. Tested in
  `test_branches.py`.

## E4: position bias (Groq qwen, 3 images, 3 shuffled orders)

Only **2 of 3 orders completed**: the third hit the `json_validate_failed` 400 (item 3 above), and
the qwen daily token cap was then reached, so it could not be re-run today.
Of the 2: rank correlation **1.0**, the same winner, every image kept its rank, and position did not
predict rank (correlation 0.0). That is evidence of no strong position bias, not proof.
Rerun `e4_position_bias.py` after the daily reset for the third order.

## E1: thumbnail size (same 3 images at 512, 384, 256 px)

| size | prompt tokens per image | mean score diff vs 512 | rank correlation vs 512 | same winner |
|---|---|---|---|---|
| 512 | 1,468 | 0 | 1.0 | yes |
| 384 | 1,468 | 3.33 | -0.5 | no |
| 256 | 1,468 | 1.0 | 1.0 | yes |

- **Image size does not change Groq's token cost** (identical at all three sizes): Groq appears to
  charge a flat amount per image. So a smaller thumbnail buys no throughput on this model, and
  512 stays.
- Agreement is not ordered by size (384 disagrees more than 256), which points to run-to-run noise
  rather than a size effect. One reading per size cannot separate the two: repeat 512 twice to
  measure the noise first. Not done: the daily token cap was reached.

## Experiments written, not run

| | what | cost | needs |
|---|---|---|---|
| E2 | prompt vs model: Klein and Nano Banana 2, baseline vs compiled prompt, 4 images, then the Critic ranks them | **$0.162** (4 paid generations; ledger would go to $0.196) | `--confirm-paid` |
| E5 | Critic calibration: `make` generates 50 free images and scores them, then you hand-score, then `report` | **$0**; 2,880 Cloudflare neurons (29% of a day) and about 80K Groq tokens (40% of qwen's daily cap) | your hand scores |
| E6 | `prompt_is_detailed` precision: the real Gate on 30 detailed and 30 vague prompts | **$0**; about 36K gpt-oss-20b tokens, about 8 minutes | nothing |
| E7 | n=4 in one call vs 4 calls, gpt-image-2 low | **about $0.048** (ledger would go to $0.082) | `--confirm-paid` |
| E3 | skipped: over the $5 cap | (about $6.30 for the `high` arm alone) | |

Run any with `PYTHONPATH=src:playground .venv/bin/python playground/<script>.py`.
