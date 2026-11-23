"""Engine errors. The frontend shows a clear message for each."""


class EngineError(Exception):
    code = "engine_error"


class ContentRefused(EngineError):
    code = "content_refused"


class OutOfCredits(EngineError):
    code = "out_of_credits"


class ProviderDown(EngineError):
    code = "provider_down"
