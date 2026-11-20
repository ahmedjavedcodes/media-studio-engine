import os

# Tests never talk to LangSmith.
os.environ["LANGSMITH_TRACING"] = "false"


import pytest

from media_engine.agents.image_grand.limiter import review_limiter


@pytest.fixture(autouse=True)
def _fresh_review_limiter():
    """The safety-review limiter is process-wide (3 per minute): every test gets a fresh one,
    or reviews in earlier tests would use up the slots of later ones."""
    review_limiter.cache_clear()
    yield
    review_limiter.cache_clear()
