from app.agents.grand.graph import build_dummy_graph


async def test_dummy_graph_runs() -> None:
    out = await build_dummy_graph().ainvoke({"prompt": "hi", "reply": ""})
    assert out["reply"] == "echo: hi"
