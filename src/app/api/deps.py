"""FastAPI dependencies for the create engine. Tests override these."""

from fastapi import Request

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.api.run_service import RunService
from app.core.tokens import TokenSigner


def get_run_service(request: Request) -> RunService:
    return request.app.state.run_service  # type: ignore[no-any-return]


def get_create_deps(request: Request) -> CreateDeps:
    return request.app.state.create_deps  # type: ignore[no-any-return]


def get_token_signer(request: Request) -> TokenSigner:
    return request.app.state.create_deps.tokens  # type: ignore[no-any-return]
