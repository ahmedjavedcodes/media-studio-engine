"""The art director's rules, checked in code. Used by the Art Director and the Prompt Writer (one
corrective retry each), the dev trace, the eval harness and the offline CI check.

The prompts tell the models the rules. This module checks that they followed them, because a model
that was told "never invent text" still does it now and then. No LLM here.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from app.agents.image_grand.workers.shot_plan import ShotPlan

MIN_FILL = 0.70  # a final prompt uses 70 to 100 percent of the model's word limit
MIN_PALETTE = 2
MIN_REALISM_CHECKS = 1

_QUOTE = re.compile(r"[\"“”«»]([^\"“”«»]+)[\"“”«»]")
_WORD = re.compile(r"[a-z'\-]+")

# Words that mean "the user asked for text or a logo". English, Roman Urdu, Urdu/Arabic script.
_TEXT_ASKED = frozenset(
    "text word words says say saying reads written caption headline title "
    "slogan tagline label labeled labelled logo logos poster banner sign signboard typography "
    "lettering letters quote name brand wordmark heading subtitle price menu likha likhna likho "
    "naam".split()
)
_TEXT_ASKED_SCRIPT = ("لکھ", "عبارت", "ٹیکسٹ", "نام", "لوگو", "نعرہ", "عنوان", "كلمة", "نص", "شعار")

_PEOPLE = frozenset(
    "person people man men woman women girl girls boy boys child children kid kids baby babies "
    "human humans lady ladies gentleman guy crowd couple family student students worker workers "
    "customer chef selfie fisherman fishermen musician baker pilot sailor villager grandmother "
    "grandfather "
    "mother father hand hands finger fingers player athlete dancer farmer doctor "
    "nurse teacher ladki ladka aadmi admi aurat bacha bachay insan shakhs larki larka".split()
)
# Left out on purpose: "silhouette", "figure", "face", "character", "model". They describe a
# bottle's outline, a design or a 3D model as often as a person, and a false alarm costs a retry.
_PEOPLE_SCRIPT = ("شخص", "آدمی", "عورت", "لڑکی", "لڑکا", "بچ", "لوگ", "انسان", "رجل", "امرأة")

# Single-word brands only when the word has no everyday meaning ("apple", "jordan", "puma" are
# left out on purpose: a fruit or an animal is not an invented brand).
_BRANDS = frozenset(
    "nike adidas samsung starbucks chanel gucci rolex tesla google disney sony dior prada "
    "mcdonalds nutella lego ferrari bmw mercedes hermes cartier tiffany ikea nintendo "
    "playstation xbox microsoft toyota audi porsche lamborghini versace burberry fendi "
    "balenciaga reebok pepsi heineken budweiser guinness marlboro kodak nikon leica gopro "
    "iphone macbook airpods".split()
)
_BRAND_PHRASES = ("coca-cola", "coca cola", "louis vuitton", "red bull", "jack daniel")

# Light described as EQUIPMENT. An image model draws what you name: "softbox" and "strip light"
# end up in the picture. Light is described by direction and quality instead. Not checked when the
# user wrote the word themselves.
_EQUIPMENT = re.compile(
    r"\b(soft ?box(es)?|octabox(es)?|strip ?lights?|reflectors?|bounce cards?|white cards?|"
    r"black cards?|foam boards?|light stands?|beauty dish(es)?|snoots?|v-flats?|c-stands?|"
    r"tripods?|light panels?|lighting rigs?)\b",
    re.IGNORECASE,
)

# Phrases that put an artefact in the picture: a bare grey studio void, a fake glow, an object that
# should rest on something but floats. Not checked when the user wrote the words themselves.
_BANNED = re.compile(
    r"\b(seamless\s+(?:light\s+|pale\s+|neutral\s+)?gr[ae]y\s+(?:background|backdrop|sweep|paper)|"
    r"glowing\s+edges?|floating)\b",
    re.IGNORECASE,
)

# Markings nobody asked for: an engraving, a monogram, a brand mark, an inscription. They put text
# or a logo on the object. Not checked when the user asked for text, a logo or a brand.
_MARKINGS = re.compile(
    r"\b(brand(ing| name| mark| engraving)?|engrav\w*|monogram\w*|emblem\w*|inscri\w+|"
    r"emboss\w*|imprint\w*|trademark\w*|wordmark)\b",
    re.IGNORECASE,
)

# Negated labels. "no label" makes a model draw a bare, unfinished object or the words "NO LABEL".
# When no text was asked for, a labelled object shows "a blank label plate" instead.
_NEGATED_LABEL = re.compile(
    r"\b(no|without)\s+(a\s+|any\s+)?(labels?|text|lettering|logos?|branding|writing)\b|"
    r"\b(unlabell?ed|unbranded|text-free|label-free)\b",
    re.IGNORECASE,
)
LABEL_PLATE = "blank label plate"
_LABEL_PLATE_RE = re.compile(r"blank\s+(label|plate|nameplate)", re.IGNORECASE)
_LABELED_OBJECTS = frozenset(
    "bottle bottles jar jars box boxes can cans tube tubes package packaging pouch bag carton "
    "tin tins sachet packet perfume cosmetic cosmetics cream serum lipstick".split()
)

_GENERIC = (
    "high detail",
    "highly detailed",
    "beautiful",
    "stunning",
    "masterpiece",
    "amazing",
    "best quality",
    "ultra realistic",
    "8k",
    "4k",
    "trending on artstation",
    "award winning",
)

_STYLE_QUESTION = re.compile(
    r"\b(style|styles|lighting|light|camera|lens|mood|colou?rs?|details?|vibe|look|feel)\b", re.I
)
NO_QUESTION_NOTE = (
    "Do not ask about style, light, camera, mood, colour or detail: decide those yourself. "
    "Leave `missing` empty and write the whole plan."
)


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def word_count(text: str) -> int:
    return len(text.split())


def word_range(max_words: int) -> tuple[int, int]:
    """Target words for a final prompt: 70 to 100 percent of the model's limit."""
    return math.ceil(max_words * MIN_FILL), max_words


def asked_for_text(prompt: str) -> bool:
    """The user asked for text, a logo or a label: a quote, a text word, or a text word in Urdu."""
    if _QUOTE.search(prompt):
        return True
    return bool(_words(prompt) & _TEXT_ASKED) or any(s in prompt for s in _TEXT_ASKED_SCRIPT)


def asked_for_people(prompt: str) -> bool:
    return bool(_words(prompt) & _PEOPLE) or any(s in prompt for s in _PEOPLE_SCRIPT)


def _mentions_brand(text: str) -> set[str]:
    low = text.lower()
    return {p for p in _BRAND_PHRASES if p in low} | (_BRANDS & _words(low))


def _ascii_prompt(prompt: str) -> bool:
    letters = [c for c in prompt if c.isalpha()]
    return sum(c.isascii() for c in letters) >= 0.9 * max(1, len(letters))


def plan_text(plan: ShotPlan) -> str:
    """Everything in the plan that can reach an image prompt, as one string."""
    parts = [
        plan.subject,
        plan.world,
        plan.moment,
        plan.composition,
        plan.depth_text(),
        plan.light_text(),
        plan.camera_text(),
        *plan.subject_details,
        *plan.props,
        *plan.materials,
        *plan.must_have,
    ]
    return " ".join(p for p in parts if p)


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str


def _artefacts(user_prompt: str, text: str, found: list[Violation]) -> None:
    """Equipment and banned phrases in `text`, unless the user asked for them."""
    gear = {m.group(0).lower() for m in _EQUIPMENT.finditer(text)}
    gear -= {m.group(0).lower() for m in _EQUIPMENT.finditer(user_prompt)}
    if gear:
        found.append(Violation("light_equipment", ", ".join(sorted(gear))))
    banned = {re.sub(r"\s+", " ", m.group(0).lower()) for m in _BANNED.finditer(text)}
    banned -= {re.sub(r"\s+", " ", m.group(0).lower()) for m in _BANNED.finditer(user_prompt)}
    if banned:
        found.append(Violation("banned_phrase", ", ".join(sorted(banned))))


def _markings(user_prompt: str, text: str, found: list[Violation]) -> None:
    if asked_for_text(user_prompt):
        return
    if marks := {m.group(0).lower() for m in _MARKINGS.finditer(text)}:
        found.append(Violation("invented_marking", ", ".join(sorted(marks))))


def _invented(user_prompt: str, text: str, found: list[Violation]) -> None:
    """People and brands in `text` that the user did not ask for. English prompts only: a word in
    another script cannot be compared with the English the models write."""
    if not _ascii_prompt(user_prompt):
        return
    if not asked_for_people(user_prompt) and (people := _words(text) & _PEOPLE):
        found.append(Violation("invented_people", ", ".join(sorted(people))))
    if brands := _mentions_brand(text) - _mentions_brand(user_prompt):
        found.append(Violation("invented_brand", ", ".join(sorted(brands))))


def check_plan(
    user_prompt: str, plan: ShotPlan, *, valid_looks: list[str] | None = None
) -> list[Violation]:
    """Every rule the ShotPlan breaks. Empty = clean. A plan that asks a question is not checked."""
    if plan.missing:
        return []
    found: list[Violation] = []
    required = {
        "subject": plan.subject,
        "intent": plan.intent.subject_type,
        "world": plan.world,
        "subject_details": plan.subject_details,
        "composition": plan.composition,
        "depth": plan.depth.mid,
        "light": plan.light.source or plan.light.direction,
        "camera": plan.camera.lens,
        "materials": plan.materials,
    }
    found += [Violation("missing_field", k) for k, v in required.items() if not v]
    if len(plan.palette) < MIN_PALETTE:
        found.append(Violation("missing_field", "palette"))
    if len(plan.realism_checks) < MIN_REALISM_CHECKS:
        found.append(Violation("missing_field", "realism_checks"))
    if valid_looks is not None and plan.look not in valid_looks:
        found.append(Violation("unknown_look", plan.look or "empty"))
    if plan.wants_text and not asked_for_text(user_prompt):
        found.append(Violation("invented_text", f"text: {plan.text.exact!r}"))
    text = plan_text(plan)
    _markings(user_prompt, text, found)
    _artefacts(user_prompt, text, found)
    _invented(user_prompt, text, found)
    return found


def check_prompt(
    user_prompt: str, plan: ShotPlan, prompt: str, *, max_words: int
) -> list[Violation]:
    """Every rule one written prompt breaks. The plan says whether text was asked for."""
    found: list[Violation] = []
    low, high = word_range(max_words)
    count = word_count(prompt)
    if count < low:
        found.append(Violation("too_short", f"{count} words, wanted {low} to {high}"))
    elif count > high:
        found.append(Violation("too_long", f"{count} words, wanted {low} to {high}"))
    for phrase in _GENERIC:
        if (
            re.search(rf"\b{re.escape(phrase)}\b", prompt, re.I)
            and phrase not in user_prompt.lower()
        ):
            found.append(Violation("generic_word", phrase))
    text_asked = asked_for_text(user_prompt) or plan.wants_text
    if not text_asked:
        if quoted := _QUOTE.findall(prompt):
            found.append(Violation("invented_text", f"quoted words: {quoted[0]!r}"))
        if re.search(r"\blogos?\b", prompt, re.I) and not _NEGATED_LABEL.search(prompt):
            found.append(Violation("invented_logo", "logo in the prompt"))
        if neg := _NEGATED_LABEL.search(prompt):
            found.append(Violation("negated_label", neg.group(0)))
        labelled = _words(f"{user_prompt} {plan.subject}") & _LABELED_OBJECTS
        if labelled and _ascii_prompt(user_prompt) and not _LABEL_PLATE_RE.search(prompt):
            found.append(Violation("label_plate_missing", f"say '{LABEL_PLATE}'"))
    if not text_asked:
        _markings(user_prompt, prompt, found)
    _artefacts(user_prompt, prompt, found)
    _invented(user_prompt, prompt, found)
    return found


def needless_question(plan: ShotPlan) -> bool:
    """The Art Director asked about style, light, camera, mood, colour or detail. It may only ask
    what the subject is, so a question like that is never shown: the model is asked again, once,
    with questions switched off."""
    if not plan.missing:
        return False
    return all(_STYLE_QUESTION.search(q.text) for q in plan.missing)


def fixable_by_retry(violations: list[Violation], *, length: bool = False) -> list[Violation]:
    """Violations a second try can fix. Length is left to the guard (trim) and the trace, except
    for the Prompt Writer, which is told to shorten (`length=True`)."""
    skip = {"too_short"} | (set() if length else {"too_long"})
    return [v for v in violations if v.code not in skip]


def ensure_label_plate(user_prompt: str, plan: ShotPlan, prompt: str) -> str:
    """Packaging with no text asked must show "a blank label plate". If the writer left it out
    after its retry, the code adds it after the first clause (never at the end, where the word
    limit trims)."""
    if asked_for_text(user_prompt) or plan.wants_text or _LABEL_PLATE_RE.search(prompt):
        return prompt
    if not _words(f"{user_prompt} {plan.subject}") & _LABELED_OBJECTS or not _ascii_prompt(
        user_prompt
    ):
        return prompt
    match = re.search(r"[,;.]", prompt)
    cut = match.start() if match else len(prompt)
    return f"{prompt[:cut]} with a {LABEL_PLATE}{prompt[cut:]}"


def sanitize_plan(user_prompt: str, plan: ShotPlan) -> ShotPlan:
    """Code fix for what code can fix: text the user never asked for. People and brands stay as
    they are (cutting them out would break the plan) and are reported by `check_plan`."""
    if plan.wants_text and not asked_for_text(user_prompt):
        return plan.model_copy(update={"text": plan.text.model_copy(update={"exact": None})})
    return plan


def sanitize_prompt(user_prompt: str, plan: ShotPlan, prompt: str) -> str:
    """Quotation marks and the words inside them are removed when no text was asked for."""
    if asked_for_text(user_prompt) or plan.wants_text or not _QUOTE.search(prompt):
        return prompt
    return re.sub(r"\s{2,}", " ", _QUOTE.sub("", prompt)).strip(" ,.")


def retry_note(violations: list[Violation]) -> str:
    """Tell the model what it broke, in plain words, for the one corrective retry."""
    notes: dict[str, Callable[[Violation], str]] = {
        "missing_field": lambda v: f"Fill the plan field `{v.detail}`.",
        "unknown_look": lambda v: f"`look` must be one id from the library, not {v.detail!r}.",
        "invented_text": lambda v: (
            "The user asked for no text. Set text.exact to null and write no quotes."
        ),
        "invented_logo": lambda v: "The user asked for no logo. Remove it.",
        "invented_people": lambda v: f"The user asked for no people. Remove: {v.detail}.",
        "invented_brand": lambda v: f"The user named no brand. Remove: {v.detail}.",
        "light_equipment": lambda v: (
            f"Describe light by direction and quality, not equipment. Remove: {v.detail}."
        ),
        "banned_phrase": lambda v: f"Remove the phrase and show the idea instead: {v.detail}.",
        "invented_marking": lambda v: (
            f"The user asked for no text, brand or markings. Remove: {v.detail}."
        ),
        "too_long": lambda v: f"Shorten each prompt ({v.detail}). Cut the least visual words.",
        "negated_label": lambda v: (
            f"Do not write {v.detail!r}. Write 'a {LABEL_PLATE}' so packaging looks finished."
        ),
        "label_plate_missing": lambda v: f"Packaging with no text asked: write 'a {LABEL_PLATE}'.",
        "generic_word": lambda v: f"Replace the generic words ({v.detail}) with visible detail.",
    }
    lines = [notes[v.code](v) for v in violations if v.code in notes]
    return " ".join(dict.fromkeys(lines))
