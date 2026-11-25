"""Build the real CreateDeps from settings. Fakes for tests are built by hand instead."""

from pathlib import Path

import httpx
from langsmith import Client

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.contracts.ports import ImageProviderPort, LLMPort, VisionPort
from app.core.config import Settings, prompts_dir
from app.core.prompts import PromptStore
from app.core.tokens import TokenSigner
from app.core.tracing import build_tracer
from app.gateway.budget import SpendGuard
from app.gateway.chains import load_chain
from app.gateway.image_gateway import ImageGateway
from app.gateway.providers.cloudflare import CloudflareClient, CloudflareVisionStandIn
from app.gateway.providers.cloudflare_create import CloudflareImages, CloudflareLLM
from app.gateway.providers.fal_images import FalImages
from app.gateway.providers.fallback import FallbackLLM, FallbackVision
from app.gateway.providers.groq import GroqLLM, GroqVision
from app.gateway.providers.openrouter_images import OpenRouterImages
from app.gateway.registry import load_registry
from app.local.fake_credits import FakeCredits
from app.local.output_store import OutputStore
from app.local.plan_store import PlanStore

_LIMITS = httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60)


def build_create_deps(s: Settings) -> tuple[CreateDeps, httpx.AsyncClient]:
    """Returns the deps and the shared HTTP client (the caller closes it on shutdown)."""
    http = httpx.AsyncClient(limits=_LIMITS)
    chain = load_chain(s)
    cf = CloudflareClient(
        http, account_id=s.cloudflare_account_id, api_token=s.cloudflare_api_token
    )
    providers: dict[str, ImageProviderPort] = {"cloudflare": CloudflareImages(cf, http)}
    if s.openrouter_images or s.fal_images:
        # One guard for every paid provider, so SPEND_CAP_USD is one budget, not one per provider.
        guard = SpendGuard(s.spend_cap_usd, ledger=Path(".cache/paid_spend.jsonl"))
        if s.openrouter_images:
            providers["openrouter"] = OpenRouterImages(
                http, api_key=s.openrouter_api_key, base_url=s.openrouter_base_url, guard=guard
            )
        if s.fal_images:
            providers["fal"] = FalImages(http, api_key=s.fal_key, guard=guard)
    gateway = ImageGateway(providers, chain.image_providers, paid=chain.paid)

    def hub_client() -> Client:
        return Client(api_key=s.langsmith_api_key)

    safety_prompt = (prompts_dir() / "guardrail" / "vision_standin.md").read_text("utf-8")
    critic: LLMPort = CloudflareLLM(cf, chain.critic.model)
    vision: VisionPort = CloudflareVisionStandIn(cf, system_prompt=safety_prompt)
    if s.groq_fallback and s.groq_api_key:  # a Cloudflare 429, 5xx or timeout asks Groq instead
        critic = FallbackLLM(
            critic, GroqLLM(http, api_key=s.groq_api_key, model=s.groq_vision_model)
        )
        vision = FallbackVision(
            vision,
            GroqVision(
                http, api_key=s.groq_api_key, model=s.groq_vision_model, system_prompt=safety_prompt
            ),
        )

    use_hub = s.prompt_source == "hub" and bool(s.langsmith_api_key)
    director_llm: LLMPort = (
        GroqLLM(http, api_key=s.groq_api_key, model=chain.enhancer.model)
        if chain.enhancer.provider == "groq"
        else CloudflareLLM(cf, chain.enhancer.model)
    )
    deps = CreateDeps(
        registry=load_registry(run_testing=s.testing_models_on),
        gateway=gateway,
        enhancer=director_llm,  # the Art Director
        writer=director_llm,  # the Prompt Writer (the same model, a lighter call)
        plans=PlanStore(s.plans_db),
        critic=critic,
        # Per-image safety stays on Cloudflare (free). OpenRouter's Llama Guard is paid: off.
        vision=vision,
        credits=FakeCredits(s.start_credits),
        store=OutputStore(Path(s.local_output_dir)),
        prompts=PromptStore(
            source="hub" if use_hub else "local",
            tag=s.hub_tag,
            refs={
                "art_director": s.art_director_prompt_ref,
                "prompt_writer": s.writer_prompt_ref,
                "critic": s.critic_prompt_ref,
            },
            client_factory=hub_client if use_hub else None,
        ),
        tracer=build_tracer(
            enabled=s.langsmith_tracing, api_key=s.langsmith_api_key, project=s.trace_project
        ),
        tokens=TokenSigner(s.stream_token_secret, ttl_s=s.stream_token_ttl_s),
        settings=s,
        enabled_providers=gateway.enabled,
    )
    return deps, http
