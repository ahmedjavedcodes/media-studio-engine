from app.core.config import Settings


def test_defaults_are_build_mode() -> None:
    s = Settings(_env_file=None)
    assert s.provider_mode == "build"
    assert s.spend_cap_usd == 2.0
