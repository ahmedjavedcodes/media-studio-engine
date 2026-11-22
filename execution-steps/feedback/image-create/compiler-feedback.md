# Compiler Spec — Review Decisions

**Scope: a separate task, own branch.** The Compiler is `gateway/compiler.py`, and
`gateway/` is off-limits to the architecture-simplify task. Keep the two apart.

## Do now (compiler + registry only)
- **1 — Style.** The registry's `nouns` for FLUX.2 is FLUX.1-era. FLUX.2 runs on a
  Mistral-3 text encoder and wants natural language — this is well-supported by Black
  Forest Labs' own current docs, so change it. Set `prompt_style`: FLUX.2 (all), Seedream
  → `sentences`; Nano Banana, GPT Image 2 stay `sentences`; Recraft → `tags`. **Qwen**:
  keep `sentences` but it's the least certain — flag it for an eval. `nouns` ends up unused;
  leave the code path, retire it from the cards. Update IMAGE-CREATE-OPTIMIZE §5.2 to match
  in this same task.
- **2 — Lead-with** (lowest priority). Add an optional `lead_order` to the card: default
  subject-first, GPT Image 2 background → subject → details. Skip if it's not cheap.
- **3 — Budgets.** Klein → 60 to match the FLUX.2 family (80 drifts). Seedream → 70 (inside
  its 30–80 range). The rest already match.
- **4 — Camera.** Don't add a schema field. The Brief leaves `camera` empty for
  illustration/design (the Brief-prompt edit already flagged), and the Compiler **omits any
  empty field**. Pair the two; no new field.
- **5 — Negatives.** Add a strip of "no X" / "without X" clauses in the Compiler, as a
  safety net under the Brief's rule.
- **8 — Final banned-word recheck.** After assembling the prompt, re-run the banned-word
  strip on the whole string, not just per field.

## Defer (separate schema / gateway decision)
- **6 — Recraft `style` + `controls.colors`.** Needs new Brief `style` and `colors` fields
  — a schema change. For now Recraft gets the short `tags` prompt with its default style.
  Decide the fields separately.
- **7 — Size / aspect ratio.** Leave size where it is (presets + gateway); don't move it
  into the Compiler. But **confirm the gateway sets size explicitly on every call** —
  especially Seedream, which defaults to 2048² and double-bills. If it doesn't, fix that in
  the gateway, not here.

## Leave as-is
- Riverflow 2.5 and Muse Image stay unverified `sentences` — don't add a template until confirmed.

**Order:** 1, 3, 4, 5, 8 first (code + registry), then 2 if cheap. 6 and 7 are their own decision.
