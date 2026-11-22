# image-create-agent — Review & Fixes

Scope: the create agent only (Brief → ImageGenModel tool → Critic, the plain pipeline).
Routing, the Grand Agent and the Designer are set aside for later.

**Verdict: the create agent is a faithful rename — logic unchanged, Brief always runs,
3 LLM calls, create envelope pinned.** One real gap, then two naming leftovers.

## Fix
- **C1 — "structure, don't pad" is not in the Brief's prompt.** We agreed the Brief
  should structure an already-detailed prompt and fill only real gaps, not inflate it.
  But `prompts/brief/brief.md` moved byte-identical (`R100`), so that guidance lives only
  in the Python docstring — the model never sees it. Check the prompt: if the
  "already-detailed → keep the user's specifics, only fill gaps" instruction isn't there,
  add it. The docstring alone changes nothing. *(prompts/brief/brief.md)*

## Minor
- **C2 — `ImageGenModel` still subclasses `Agent`.** The spec framed it as a tool, and it
  now lives in `tools/`, but it's still an `Agent` subclass (rename-only, as decided). It
  has no LLM, so this is harmless — just note it isn't fully a plain tool yet, and finish
  that conversion when convenient. *(tools/image_gen_model.py)*

## Step 8 (leftovers — keep tracked)
- **C3** — the create agent is still built with the config key `image_grand_agent`
  (`AgentConfig.from_policy("image_grand_agent", …)`). Kept on purpose (R-5, it's a stored
  string), but misleading now. Rename it with the stored `PlanStep.agent` strings in one
  coordinated change.
- **C4** — `schemas/agent_io.py:31` still says `prompt_is_detailed` "will skip the Brief".
  That's now wrong (the Brief always runs). Fix the comment.

## Correct — don't touch
- Brief always runs, proven by the `fast_llm, reasoning, vision` test on a detailed prompt.
- First-pass create is exactly 3 LLM calls; the runner adds none.
- Create returns a document with one image layer; the output shape is pinned by a test.
- `NEEDS_INPUT` from the Brief passes up unchanged.
- Brief, ImageGenModel and Critic moves are import/name-only; prompt files are byte-identical.

*(Deferred to later: the Designer trigger, the design stub-fallback, the complex-prompt
UX, and the early-exit envelope shapes — all routing/Grand-level, not the create agent.)*
