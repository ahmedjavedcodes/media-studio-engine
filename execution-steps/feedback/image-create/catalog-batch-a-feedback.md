# Catalog Batch (a) — Review

**Verdict: approve after one fix (F1).** 751 passing, lock == gateway-check is exactly
right, the offered/available split is clean. Fix F1, then commit (a).

## Fix before moving on
- **F1 — the dev shim never threads `plan` into `auth`.** The UI sends `plan` and the cap
  is now per-plan, but `/runs` builds `auth` from the default (free), so **every run executes
  as free** — a "pro" user is still capped at 2 and gated as free at runtime. Only `/catalog`
  respects the plan. Fix: add `plan` to `CreateBody` and build `AuthContext(plan=...)` from
  it in `execute()`. Add a test: pro n=4 succeeds, free n=4 refused end-to-end through the
  shim. Without this, decision 6 doesn't actually vary by plan, and batch (b)'s per-plan
  model gating won't be testable either.

## Answers to your questions
- **1 — offered vs available: yes, that's the intended meaning.** `offered: false` =
  registered and still callable if a chain names it, but not listed and not user-pickable.
  Nano Banana 2.1 sits dormant until priced — correct.
- **4 — Basic resolution: accept 2K, drop "~1K" from the spec.** 2K is honestly reachable (a
  free user who picks a Basic paid model like Seedream Flash), and the default free path
  (flux-1-schnell) is already ≤1024 via the presets. A hard 1K free cap would be a new
  `setting_policy` rule and might break Seedream Flash's min size — only add it later if you
  want free strictly cost-capped. I'll update the spec to 2K.
- **GPT Image 2 (high) "batch only": add a `batch_only: bool` flag now** (cheap). Today it's
  just an Ultra option with no guard, so batch (b) could put it on the interactive path.
  Flag it so the picker/gateway keep it off the fast path.

## Nits (optional)
- **F2** — over-cap rejection has two shapes: n>4 → 422, free n=3 → FAILED run. The real
  backend should return one validation-error shape. Shim-only, low priority.
- **F3** — an `offered: false`, unpriced model could break budget if a chain ever names it
  (price is None). None does today; add a guard or just note it.
- **Slash no-op** — "/edit" with no text after it leaves Generate enabled but does nothing
  (`parseSlash` strips it, text is empty, `generate()` returns). Disable on the stripped text.

## Right — don't touch
- Lock = `plan_block`, the same check the gateway enforces with, proven equal by a test.
- Two-flag design (available vs offered), derived `max_resolution`, the preset size math
  (÷16, ≤1024, within tolerance), defence-in-depth cap, and the `uncapped()` branch-test
  helper are all sound.
