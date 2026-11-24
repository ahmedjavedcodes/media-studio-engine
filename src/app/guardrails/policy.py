"""Tiers, thresholds and the Llama Guard S-code map. See spec sections 7.4, 8, 9."""

from app.guardrails.schemas import Category, Decision, Tier

# Decision thresholds (section 8)
HARD_THRESHOLD = 0.85
SOFT_THRESHOLD = 0.60

# Input limits (sections 7.1, 7.3)
MAX_PROMPT_CHARS = 4000
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_IMAGES_PER_REQUEST = 10  # not in the spec: stops a request from flooding the vision call
IMAGES_PER_CALL = 5
IMAGE_LONG_SIDE = 768
JPEG_QUALITY = 90

# Timeouts in seconds, per PROVIDER_MODE (GUARDRAIL-EVAL-REVIEW.md, fix 2).
# Jev is the same in both modes. A retry delay of None means no retry.
JEV_TIMEOUT_S = 0.8
JEV_RETRY_DELAY_S = 0.4
# build: Cloudflare Scout stand-in
BUILD_VISION_TIMEOUT_S = 2.5
BUILD_VISION_RETRY_DELAY_S: float | None = None
BUILD_DEADLINE_TEXT_S = 1.5
BUILD_DEADLINE_IMAGES_S = 3.0
# test: Llama Guard 4. Placeholders until real latency is measured.
TEST_VISION_TIMEOUT_S = 1.5
TEST_VISION_RETRY_DELAY_S: float | None = 0.8
TEST_DEADLINE_TEXT_S = 2.0
TEST_DEADLINE_IMAGES_S = 2.0
MIN_RETRY_WINDOW_S = 0.1

# Cooldown rules that are not configurable. Strike limits and the 1/2 minute cooldowns
# live in core/config.py.
REJECT_WINDOW_MINUTES = 10
REJECTS_FOR_COOLDOWN = 5
LONG_COOLDOWN_SEC = 30 * 60

CATEGORY_TIER: dict[Category, Tier] = {
    Category.SAFE: Tier.NONE,
    Category.CSAM: Tier.CRITICAL,
    Category.TERRORISM_WEAPONS: Tier.CRITICAL,
    Category.NCII_UNDRESSING: Tier.SEVERE,
    Category.GORE_SELF_HARM: Tier.SEVERE,
    Category.NSFW_EXPLICIT: Tier.STANDARD,
    Category.HATE_HARASSMENT: Tier.STANDARD,
    Category.CELEBRITY_LIKENESS: Tier.STANDARD,
    Category.PROMPT_INJECTION: Tier.STANDARD,
    Category.PII_WATERMARK: Tier.STANDARD,
    Category.OTHER_UNSAFE: Tier.STANDARD,
}

# Llama Guard 4 hazard codes -> our category
S_CODE_CATEGORY: dict[str, Category] = {
    "S4": Category.CSAM,
    "S9": Category.TERRORISM_WEAPONS,
    "S3": Category.NCII_UNDRESSING,
    "S11": Category.GORE_SELF_HARM,
    "S12": Category.NSFW_EXPLICIT,
    "S10": Category.HATE_HARASSMENT,
    "S5": Category.CELEBRITY_LIKENESS,
    "S7": Category.PII_WATERMARK,
    "S8": Category.PII_WATERMARK,
    "S1": Category.OTHER_UNSAFE,
    "S2": Category.OTHER_UNSAFE,
    "S6": Category.OTHER_UNSAFE,
    "S13": Category.OTHER_UNSAFE,
    "S14": Category.OTHER_UNSAFE,
}

_TIER_RANK = {Tier.NONE: 0, Tier.STANDARD: 1, Tier.SEVERE: 2, Tier.CRITICAL: 3}
_DECISION_RANK = {Decision.APPROVE: 0, Decision.STANDARD_REJECT: 1, Decision.INSTANT_REJECT: 2}


def tier_of(category: Category) -> Tier:
    return CATEGORY_TIER[category]


def tier_rank(tier: Tier) -> int:
    return _TIER_RANK[tier]


def decision_rank(decision: Decision) -> int:
    return _DECISION_RANK[decision]


def decide(category: Category, p: float) -> Decision:
    """Section 8: turn a violation probability into a decision."""
    tier = tier_of(category)
    if tier is Tier.NONE or p < SOFT_THRESHOLD:
        return Decision.APPROVE
    if tier in (Tier.CRITICAL, Tier.SEVERE) and p >= HARD_THRESHOLD:
        return Decision.INSTANT_REJECT
    return Decision.STANDARD_REJECT


def combo_scores(scores: dict[str, float]) -> dict[Category, float]:
    """Combo rules over Jev question scores (keys are question ids in jev_questions.yaml).

    CSAM: the request involves a minor AND asks for sexual or undressing content.
    Each side is its own question. Both must be high, so the combo score is the lower one.
    """
    minor = scores.get("involves_minor", 0.0)
    sexual = max(
        scores.get("sexual_content", 0.0),
        scores.get("undress_real_person", 0.0),
        scores.get("explicit_face_swap", 0.0),
    )
    return {Category.CSAM: min(minor, sexual)}
