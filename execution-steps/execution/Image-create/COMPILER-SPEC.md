# Compiler Spec (gateway/compiler.py)

The step between the Brief and the image model. Plain code, no LLM, no cost, deterministic
(same Brief + model → same prompt). Reads the model's registry card (`prompt_style`,
`prompt_max_words`, `billing_unit`, `supported_sizes`), so nothing is hardcoded per model.

**In:** Brief fields + chosen model. **Out:** one prompt string in that model's style,
within budget, plus model params (size, style, colors).

## Universal rules
- Strip banned words: `pristine, subtle, hints of, fading, muted, soft diffused, gentle,
  minimal, evoking, atmosphere, balanced, serene`, and any trailing mood clause.
- Keep it concrete: countable nouns, a light direction, ≥ 2 `detail_anchors`.
- No negatives ("no X"). Describe what is present.
- Set size explicitly every call (Seedream defaults to 2048² and double-bills). Aspect
  ratio comes from the Brief.
- No text in the prompt — `text_wanted` is a designer layer; describe a clean area instead.
- Re-run the blocklist on the output before returning.

## Two shapes

**A — natural-language narrative** (FLUX.2, Nano Banana, GPT Image 2, Seedream, Qwen).
Fields → sentences: `subject`+`action` open, then `setting`, `lighting`, `camera`
(photographic only), `detail_anchors` close.

| Model | Lead with | Budget | Note |
|---|---|---|---|
| FLUX.2 (Klein/Pro/Flex/Max) | subject | ≤ 60 words | per-MP; hex colors OK |
| Seedream 5.0 (Flash/Lite) | subject | 30–80 words | subject · composition · lighting · style clauses; name the light |
| Qwen Image 3 | subject | ≤ 60 words | long OK; strongest for text |
| Nano Banana 2 / Lite | subject | ≤ 50 words | token-billed, keep tight |
| GPT Image 2 | background → subject → details | ≤ 40 words | token-billed, shortest; start weighs most |

**B — short prompt + parameters** (Recraft). Prompt ≤ 30 words (`subject`+`action`+
`setting`+ one detail); style → `style` param; colors → `controls.colors`. Best for logos,
icons, flat illustration, vector.

## Notes
- `camera` only for photographic subjects; drop it for illustration/logo/icon.
- Riverflow 2.5 and Muse Image: no verified template yet — confirm before adding a row.
- Product-ad effects (ice, garnish) are a later Brief/supervisor job, not the Compiler.
