"""Guess the subject type of a raw prompt, in code, before any model reads it.

Used to find similar good plans (the plan store) and to group the eval. The Art Director decides
the real `intent.subject_type` itself. This guess only has to be right often enough to narrow the
search, and it reads English, Roman Urdu and Urdu words. Anything it does not know is "other".
"""

import re

_TOKENS = re.compile(r"[a-z]+|[؀-ۿ]+")

# Checked in this order: the first group with a hit wins.
_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "poster",
        frozenset(
            "poster banner flyer logo signboard sign card headline slogan typography lettering "
            "cover likha likhna likho پوسٹر لوگو لکھا".split()
        ),
    ),
    (
        "food",
        frozenset(
            "food dish meal ramen pizza burger cake dessert soup salad bread breakfast sushi pasta "
            "cheeseburger fries biryani kebab khana کھانا بریانی".split()
        ),
    ),
    (
        "product",
        frozenset(
            "product bottle perfume shoe shoes sneaker headphones watch wallet jar packaging cup "
            "mug phone laptop ad advert commercial botal jootay بوتل پرفیوم جوتا جوتے".split()
        ),
    ),
    (
        "person",
        frozenset(
            "person people man men woman women girl boy child children kid portrait selfie couple "
            "family chef musician farmer fisherman ladki ladka larki larka aurat aadmi admi bacha "
            "لڑکی لڑکا عورت آدمی بچہ بچی".split()
        ),
    ),
    (
        "animal",
        frozenset(
            "cat dog fox bird horse lion tiger elephant fish wolf bear rabbit deer owl eagle "
            "butterfly animal billi kutta بلی کتا پرندہ شیر گھوڑا".split()
        ),
    ),
    (
        "fantasy",
        frozenset(
            "dragon wizard magic castle fairy elf unicorn spaceship alien robot mythical monster "
            "ghost sci-fi jadui jadu جادو".split()
        ),
    ),
    (
        "abstract",
        frozenset(
            "abstract pattern geometric gradient texture fractal shapes wallpaper swirl "
            "پیٹرن".split()
        ),
    ),
    (
        "place",
        frozenset(
            "mountain mountains lake beach forest city street sunset sunrise landscape building "
            "architecture interior room village harbour town river desert valley bridge road pahar "
            "pahari jheel samandar manzar پہاڑ پہاڑوں جھیل سمندر منظر ساحل".split()
        ),
    ),
)


def guess_subject_type(prompt: str) -> str:
    """One of the eight subject types, or "other"."""
    if '"' in prompt or "“" in prompt:
        return "poster"  # a quoted phrase means words in the image
    words = set(_TOKENS.findall(prompt.lower()))
    words |= {w[:-1] for w in words if w.endswith("s") and len(w) > 3}  # rabbits -> rabbit
    for name, vocabulary in _GROUPS:
        if words & vocabulary:
            return name
    return "other"
