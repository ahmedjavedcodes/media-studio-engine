"""The eval harness, the golden set, the offline CI check of the art-director rules, and the A/B
tool. No provider is called."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from evals import ab_compare as ab
from evals import txt2img_eval as ev

from app.agents.image_grand.workers import art_director as ad
from app.agents.image_grand.workers import plan_rules as rules
from app.agents.image_grand.workers import prompt_writer as pw
from app.agents.image_grand.workers.shot_plan import ShotPlan
from app.agents.image_grand.workers.subject_type import guess_subject_type
from app.core.plans import Plan
from app.gateway.looks import get_looks
from app.gateway.registry import load_registry
from app.gateway.router import classify_prompt
from app.local import plan_store
from tests.unit.create_helpers import FakeLLM, FakeProvider, critic_ok, jpeg, make_deps

REG = load_registry()
SEEDS = {s["id"]: s for s in json.loads(plan_store.SEEDS.read_text("utf-8"))}


# ---- the golden dataset -------------------------------------------------------------------------


def test_the_golden_set_has_60_prompts_across_eight_subject_types_with_urdu_and_roman_urdu() -> (
    None
):
    data = ev.load_dataset()
    assert len(data) == 60 and len({e["id"] for e in data}) == 60
    types = Counter(e["subject_type"] for e in data)
    assert set(types) == {
        "product",
        "person",
        "animal",
        "place",
        "food",
        "fantasy",
        "poster",
        "abstract",
    }
    assert all(7 <= n <= 8 for n in types.values())
    langs = Counter(e["lang"] for e in data)
    assert langs["roman_ur"] >= 10 and langs["ur"] >= 6 and langs["en"] >= 36
    for t in types:  # every subject type has Urdu or Roman Urdu prompts
        assert {e["lang"] for e in data if e["subject_type"] == t} >= {"en", "roman_ur"}
    assert all(e["human_rating"] is None for e in data)  # the slot is there and empty


def test_golden_prompts_match_their_subject_type_and_router_category() -> None:
    data = ev.load_dataset()
    assert [e["id"] for e in data if guess_subject_type(e["prompt"]) != e["subject_type"]] == []
    assert [e["id"] for e in data if classify_prompt(e["prompt"]) != e["category"]] == []


def test_no_golden_prompt_is_also_a_seed_prompt() -> None:
    seeds = {s["prompt"].lower() for s in SEEDS.values()}
    assert not [e["id"] for e in ev.load_dataset() if e["prompt"].lower() in seeds]


# ---- the offline CI check: ten fixed prompts meet the rules -------------------------------------

FIXED = [
    "product-perfume", "product-sneaker", "person-fisherman", "animal-fox", "place-lake",
    "food-ramen", "fantasy-dragon", "poster-coffee", "abstract-waves", "person-reader",
]  # fmt: skip


@pytest.mark.parametrize("seed_id", FIXED)
async def test_the_director_and_writer_output_meet_the_rules_for_ten_fixed_prompts(
    seed_id: str,
) -> None:
    seed = SEEDS[seed_id]
    looks = get_looks()
    llm = FakeLLM("fake", [ShotPlan.model_validate(seed["plan"])])
    report: dict[str, Any] = {}
    plan = await ad.direct(
        llm, base_prompt="B", prompt=seed["prompt"], aspect_ratio="1:1", resolution="1K",
        answers={}, allow_questions=True, n=1, looks=looks, max_tokens=2500, timeout=5,
        report=report,
    )  # fmt: skip
    assert len(llm.calls) == 1 and report["violations"] == []  # no corrective retry was needed
    assert plan.look in looks.valid_ids() and 2 <= len(plan.palette) <= 3 and len(plan.props) <= 3
    assert plan.depth.mid and plan.light_text() and plan.camera.lens and plan.realism_checks
    assert plan.wants_text == rules.asked_for_text(seed["prompt"])  # no invented text, none lost
    for c in (REG.get("seedream-5-flash"), REG.get("flux-2-klein-4b"), REG.get("recraft-v4-1")):
        assert c is not None
        text = pw.render_template(c, plan)
        found = {v.code for v in rules.check_prompt(seed["prompt"], plan, text, max_words=999)}
        found -= {"too_short", "too_long"}  # the guard trims length. Every other rule must hold
        assert found == set(), (c.id, found)  # the code renderer obeys every rule too


def test_the_check_has_teeth_a_bad_plan_fails_it() -> None:
    bad = ShotPlan.model_validate({**SEEDS["product-perfume"]["plan"], "props": ["a woman's hand"]})
    found = {v.code for v in rules.check_plan("perfume bottle on marble", bad)}
    assert "invented_people" in found


# ---- cells and cost -----------------------------------------------------------------------------


def test_cells_cover_prompts_times_models_and_can_be_narrowed() -> None:
    cards = ev.curated_models(REG)
    assert len(cards) == 14 and all(c.picker for c in cards)
    cells = ev.make_cells(ev.load_dataset(), cards)
    assert len(cells) == 60 * 14
    small = ev.make_cells(ev.load_dataset(), cards[:2], categories=("product",), per_category=3)
    assert len(small) == 3 * 2 and {c.category for c in small} == {"product"}


def test_the_full_sweep_estimate_is_over_the_cap_and_a_small_run_is_under() -> None:
    reg = load_registry(run_testing=True)
    enabled, paid = ["openrouter", "cloudflare"], {"openrouter", "fal"}
    full = ev.plan_run(ev.make_cells(ev.load_dataset(), ev.curated_models(reg)), reg, enabled, paid)
    assert full.estimate_usd > 2.0 and not full.skipped_models
    two = [reg.get("seedream-5-flash"), reg.get("seedream-5-pro")]
    small = ev.plan_run(
        ev.make_cells(ev.load_dataset(), two, categories=("product",), per_category=1),  # type: ignore[arg-type]
        reg,
        enabled,
        paid,
    )
    assert small.estimate_usd == pytest.approx(0.018 + 0.045)


def test_models_that_cannot_run_are_skipped_and_cost_nothing() -> None:
    cells = ev.make_cells(ev.load_dataset(), [REG.get("seedream-5-flash")], per_category=1)  # type: ignore[list-item]
    plan = ev.plan_run(cells, REG, ["cloudflare"], {"openrouter"})  # OpenRouter is off
    assert plan.skipped_models == ["seedream-5-flash"] and plan.estimate_usd == 0 and not plan.cells


def test_gpt_image_is_estimated_at_its_medium_price() -> None:
    reg = load_registry(run_testing=True)
    flare = reg.get("gpt-image-2-5-flare")
    assert flare is not None
    assert ev.cell_cost_usd(flare, ["openrouter"], {"openrouter"}) == 0.03  # Pro = medium


# ---- one cell, through the real graph with fake providers ---------------------------------------


async def test_a_cell_runs_the_graph_and_keeps_scores_rules_and_a_rating_slot(
    tmp_path: Path,
) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    cell = ev.Cell("product-01", "product", "en", "perfume bottle on marble", "flux-2-klein-4b")
    row = await ev.run_cell(deps, cell, images_dir=tmp_path / "imgs")
    assert row["status"] == "ok" and row["model_id"] == "flux-2-klein-4b"
    assert row["critic"]["mean"] == pytest.approx(7.5) and row["critic"]["match"] == 7
    assert row["human_rating"] is None and row["cost_usd"] == 0.001
    assert row["content_category"] == "product" and row["look"] == "lifestyle"
    assert row["subject_type"] == "product" and row["final_prompt"]
    assert (tmp_path / "imgs").exists() and any((tmp_path / "imgs").iterdir())
    assert isinstance(row["rule_violations"], list)


async def test_a_failing_cell_is_a_row_not_a_crash(tmp_path: Path) -> None:
    deps = make_deps(tmp_path)

    class AllFail(FakeProvider):
        async def generate(self, **kw: Any) -> Any:
            from app.core.errors import ProviderCallFailed

            raise ProviderCallFailed("boom", retryable=False)

    deps.gateway._providers["cloudflare"] = AllFail()  # type: ignore[assignment]
    cell = ev.Cell("p", "product", "en", "a cup", "flux-2-klein-4b")
    row = await ev.run_cell(deps, cell)
    assert row["status"] in ("error", "no_image")


async def test_the_run_stops_at_the_cap(tmp_path: Path) -> None:
    class Dear(FakeProvider):
        async def generate(self, **kw: Any) -> Any:
            from app.contracts.ports import GeneratedImage

            return GeneratedImage(data=self.data, cost_usd=0.5)  # every image bills 0.5

    deps = make_deps(tmp_path, critic=[critic_ok(1)], provider=Dear())
    deps.registry.get("flux-2-klein-4b").providers[0].cost = 0.5  # type: ignore[union-attr]
    deps.gateway._paid = {"cloudflare"}  # pretend this provider is paid  # noqa: SLF001
    cells = [ev.Cell(f"p{i}", "general", "en", "a red fox", "flux-2-klein-4b") for i in range(5)]
    out_file = tmp_path / "rows.jsonl"
    rows = await ev.run_eval(
        deps, cells, cap_usd=1.2, out_file=out_file, concurrency=1, say=lambda _: None
    )
    statuses = Counter(r["status"] for r in rows)
    assert (
        statuses["ok"] == 2 and statuses["skipped_cap"] == 3
    )  # 2 x 0.5 fits 1.2, a third does not
    assert len(ev.load_rows(out_file)) == statuses["ok"]  # only real runs reach the file


# ---- the report, the rank file and the router ---------------------------------------------------


def make_rows(scores: dict[str, dict[str, float]], per: int = 3) -> list[dict[str, Any]]:
    rows = []
    for model, by_cat in scores.items():
        for cat, mean in by_cat.items():
            for i in range(per):
                rows.append(
                    {
                        "prompt_id": f"{cat}-{i}",
                        "category": cat,
                        "model_id": model,
                        "status": "ok",
                        "critic": {"mean": mean},
                        "cost_usd": REG.get(model).cost_per_image,  # type: ignore[union-attr]
                        "human_rating": None,
                    }
                )
    return rows


def test_scores_blend_the_human_rating_in_and_ignore_failures() -> None:
    row = {"prompt_id": "a", "model_id": "m", "status": "ok", "critic": {"mean": 6.0}}
    assert ev.cell_score(row, {}) == 6.0
    assert ev.cell_score(row, {"a|m": 5}) == pytest.approx(8.0)  # 0.5*6 + 0.5*(5*2)
    assert ev.cell_score({**row, "status": "error"}, {}) is None
    assert ev.cell_score({**row, "human_rating": 1}, {}) == pytest.approx(4.0)


def test_winners_are_per_category_and_per_plan_and_respect_the_tier() -> None:
    rows = make_rows(
        {
            "flux-2-klein-4b": {"product": 6.0},  # Basic tier
            "seedream-5-flash": {"product": 7.0},  # Pro tier
            "nano-banana-pro": {"product": 9.0},  # Max tier
        }
    )
    table = ev.score_table(rows)
    ranks = ev.winners(table, REG)
    assert ranks["product"]["basic"] == ["seedream-5-flash", "flux-2-klein-4b"]  # no Max model
    assert ranks["product"]["max"][0] == "nano-banana-pro"
    md = ev.render_table(ranks, table)
    assert "| product |" in md and "nano-banana-pro (9.0)" in md


def test_a_model_with_too_few_scored_cells_is_not_ranked() -> None:
    table = ev.score_table(make_rows({"seedream-5-flash": {"product": 9.0}}, per=2))
    assert ev.winners(table, REG) == {}


def test_equal_scores_go_to_the_cheaper_model() -> None:
    table = ev.score_table(
        make_rows({"seedream-5-flash": {"photo": 7.0}, "flux-2-klein-4b": {"photo": 7.0}})
    )
    assert ev.winners(table, REG)["photo"]["pro"][0] == "flux-2-klein-4b"


def test_write_auto_rank_replaces_only_its_own_block_and_the_router_reads_it(
    tmp_path: Path,
) -> None:
    src = ev.MODELS_YAML.read_text("utf-8")
    f = tmp_path / "models.yaml"
    f.write_text(src)
    ranks = {"product": {"pro": ["seedream-5-flash"], "max": ["flux-3", "seedream-5-pro"]}}
    ev.write_auto_rank(f, ranks)
    first = f.read_text("utf-8")
    assert first.startswith(src.rstrip("\n")[:200]) and first.count(ev.RANK_BEGIN) == 1
    ev.write_auto_rank(f, {"photo": {"max": ["flux-3"]}})  # a second write replaces the block
    second = f.read_text("utf-8")
    assert second.count(ev.RANK_BEGIN) == 1 and "product:" not in second.split(ev.RANK_BEGIN)[1]
    reg = load_registry(f)
    assert reg.ranked("photo", Plan.MAX) == ["flux-3"] and reg.ranked("product", Plan.MAX) == []
    assert second.split(ev.RANK_BEGIN)[0].rstrip("\n") == src.rstrip("\n")  # the rest untouched


def test_write_auto_rank_refuses_an_unknown_model_and_leaves_the_file(tmp_path: Path) -> None:
    f = tmp_path / "models.yaml"
    f.write_text(ev.MODELS_YAML.read_text("utf-8"))
    before = f.read_text("utf-8")
    with pytest.raises(Exception, match="unknown model"):
        ev.write_auto_rank(f, {"product": {"pro": ["no-such-model"]}})
    assert f.read_text("utf-8") == before


def test_the_real_models_yaml_has_no_stale_rank() -> None:
    assert isinstance(load_registry().auto_rank, dict)  # the shipped file loads (empty until run)


class FakeLangSmith:
    def __init__(self, exists: bool) -> None:
        self.exists = exists
        self.created: list[Any] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return self.exists

    def create_dataset(self, name: str, **kw: Any) -> Any:
        self.created.append(("dataset", name))
        return type("D", (), {"id": "d1"})()

    def create_examples(self, **kw: Any) -> None:
        self.created.append(("examples", len(kw["examples"])))


def test_push_dataset_creates_once_with_60_examples() -> None:
    client = FakeLangSmith(exists=False)
    assert ev.push_dataset(client, ev.load_dataset()) == "created"
    assert client.created == [("dataset", "txt2img-golden"), ("examples", 60)]
    again = FakeLangSmith(exists=True)
    assert ev.push_dataset(again, ev.load_dataset()) == "exists" and again.created == []


# ---- the aesthetic report and the shared plan ---------------------------------------------------


def row(model: str, look: str, stype: str, aes: int | None, real: int = 8) -> dict[str, Any]:
    crit = {"mean": 7.0, "aesthetic": aes, "realism": real}
    return {
        "prompt_id": "p",
        "model_id": model,
        "look": look,
        "subject_type": stype,
        "status": "ok",
        "critic": crit,
        "category": "product",
    }


def test_aesthetic_is_averaged_per_look_subject_type_and_model() -> None:
    rows = [
        row("a", "lifestyle", "product", 8), row("a", "lifestyle", "product", 6, 6),
        row("b", "studio", "product", 4),
        row("b", "studio", "animal", None),  # no aesthetic: skipped
        {**row("b", "studio", "animal", 9), "status": "error"},  # not scored: skipped
    ]  # fmt: skip
    rep = ev.aesthetic_report(rows)
    assert rep["look"]["lifestyle"] == {"aesthetic": 7.0, "realism": 7.0, "n": 2}
    assert rep["look"]["studio"]["aesthetic"] == 4.0 and "animal" not in rep["subject_type"]
    assert rep["model"]["a"]["n"] == 2
    md = ev.render_aesthetic(rep)
    assert "| lifestyle | 7.00 | 7.00 | 2 |" in md and md.index("lifestyle") < md.index("studio")


async def test_one_plan_per_prompt_is_shared_by_every_model(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    cells = [
        ev.Cell("p1", "product", "en", "perfume bottle on marble", m, "product")
        for m in ("flux-2-klein-4b", "flux-2-klein-9b")
    ]
    rows = await ev.run_eval(
        deps, cells, cap_usd=1.0, out_file=tmp_path / "r.jsonl", concurrency=2, say=lambda _: None
    )
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert isinstance(deps.enhancer, ev.CachingLLM)
    assert len(deps.enhancer._inner.calls) == 1  # type: ignore[attr-defined]  # planned once, used twice
    assert rows[0]["look"] == rows[1]["look"]


async def test_a_failed_plan_is_not_cached() -> None:
    from app.core.errors import ProviderCallFailed

    inner = FakeLLM("m", [ProviderCallFailed("429", retryable=True), ShotPlan(subject="x")])
    cache = ev.CachingLLM(inner)
    kw: dict[str, Any] = {"system": "s", "user": "u", "schema": ShotPlan}
    with pytest.raises(ProviderCallFailed):
        await cache.complete_json(**kw)
    first = await cache.complete_json(**kw)
    second = await cache.complete_json(**kw)
    assert first.subject == second.subject == "x" and len(inner.calls) == 2 and first is not second


# ---- the A/B tool -------------------------------------------------------------------------------


def test_the_ab_estimate_counts_every_paid_arm() -> None:
    reg = load_registry(run_testing=True)
    cards = [reg.get("seedream-5-flash"), reg.get("nano-banana-2-lite")]
    est = ab.estimate_usd(
        cards, ["openrouter", "cloudflare"], {"openrouter"}, ["pipeline", "raw", "final_raw"]
    )  # type: ignore[arg-type]
    assert est == pytest.approx(3 * (0.018 + 0.0336))
    assert (
        ab.estimate_usd([reg.get("flux-2-klein-4b")], ["cloudflare"], {"openrouter"}, list(ab.ARMS))
        == 0
    )  # type: ignore[list-item]


async def test_ab_makes_the_arms_scores_them_together_and_saves_a_side_by_side(
    tmp_path: Path,
) -> None:
    from app.agents.image_grand.workers.critic import CriticImage, CriticOutput, Scores

    sc = lambda a: Scores(match=8, craft=8, interest=8, technical=8, aesthetic=a, realism=8)  # noqa: E731
    out = CriticOutput(
        images=[CriticImage(id=f"img_{i}", scores=sc(9 - i)) for i in (1, 2, 3)], highlight="x"
    )
    provider = FakeProvider(data=jpeg((96, 96)))
    deps = make_deps(tmp_path / "d", provider=provider, critic=[critic_ok(1), out])
    folder = tmp_path / "ab"
    card = deps.registry.get("flux-2-klein-4b")
    results = await ab.run_ab(
        deps,
        "perfume bottle on marble",
        [card],
        out_dir=folder,
        arms=("pipeline", "raw", "final_raw"),
        say=lambda _: None,
    )  # type: ignore[list-item]
    assert [r.arm for r in results] == ["pipeline", "raw", "final_raw"] and all(
        r.status == "ok" for r in results
    )
    assert results[1].prompt_sent == "perfume bottle on marble"  # raw: the user's words, untouched
    assert (
        results[2].prompt_sent == results[0].prompt_sent
        and results[0].prompt_sent != results[1].prompt_sent
    )
    assert [r.scores["aesthetic"] for r in results if r.scores] == [8, 7, 6]  # one call, relative
    raw_call = [c for c in provider.calls if c["prompt"] == "perfume bottle on marble"]
    assert len(raw_call) == 1 and provider.calls[-1]["prompt"] == results[0].prompt_sent
    report = ab.write_report(results, folder, "perfume bottle on marble")
    assert (
        report.exists()
        and (folder / "side_by_side.png").exists()
        and (folder / "report.json").exists()
    )
    assert (folder / "flux-2-klein-4b__pipeline.trace.json").exists()  # the trace is kept
    trace = json.loads((folder / "flux-2-klein-4b__pipeline.trace.json").read_text())
    assert any(sp["name"] == "art_director" for sp in trace["spans"])
