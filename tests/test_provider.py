"""Smoke tests for the agentmemory Hermes plugin.

These tests exercise the provider WITHOUT booting the full Hermes Agent and
WITHOUT calling Anthropic. They verify shape, lifecycle, and tool dispatch.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


# ---- shim the `agent.memory_provider` import so we can run standalone -----

if "agent" not in sys.modules:
    fake_agent = types.ModuleType("agent")
    fake_mp = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # minimal stand-in mirroring the real ABC
        def __init__(self):
            pass

    fake_mp.MemoryProvider = MemoryProvider
    sys.modules["agent"] = fake_agent
    sys.modules["agent.memory_provider"] = fake_mp


# Load the plugin directory as a package so the relative import inside
# __init__.py (`from .agentmemory_py import ...`) resolves correctly.
import importlib.util

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR.parent))
spec = importlib.util.spec_from_file_location(
    "agentmemory_plugin",
    PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["agentmemory_plugin"] = mod
spec.loader.exec_module(mod)  # type: ignore


def _new_provider(tmp_path: Path) -> "mod.AgentMemoryProvider":
    p = mod.AgentMemoryProvider()
    # patch the summarizer to a stub so we don't need an Anthropic key
    p._store = mod.EpisodicStore()
    p.initialize("session-1", hermes_home=str(tmp_path))
    p._summarizer = mod.OnDemandSummarizer(
        llm=lambda prompt: f"STUB-SUMMARY({len(prompt)} chars)",
        max_tokens=200,
    )
    return p


def test_lifecycle_writes_then_recalls(tmp_path):
    p = _new_provider(tmp_path)

    p.sync_turn("I prefer Postgres", "Noted: you prefer Postgres.")
    p.sync_turn("the project uses MongoDB now", "Updated: project uses MongoDB.")

    summary = p.prefetch("what database does the user use?")
    assert summary.startswith("STUB-SUMMARY"), summary

    trace = (tmp_path / "agentmemory" / "trace.jsonl").read_text().strip().splitlines()
    assert len(trace) == 1
    record = json.loads(trace[0])
    assert record["intent"] == "what database does the user use?"
    assert len(record["event_ids"]) >= 2


def test_recall_tool(tmp_path):
    p = _new_provider(tmp_path)
    p.sync_turn("hello", "hi")
    out = p.handle_tool_call("agentmemory_recall", {"query": "hi"})
    parsed = json.loads(out)
    assert "summary" in parsed
    assert "event_ids" in parsed


def test_forget_is_real(tmp_path):
    p = _new_provider(tmp_path)
    p.sync_turn("ephemeral", "noted")
    eid = p._store._events[0].id
    out = p.handle_tool_call("agentmemory_forget", {"event_id": eid})
    assert json.loads(out)["removed"] == 1
    # the event id should be GONE — no tombstone
    assert p._store.get(eid) is None


def test_drift_tool(tmp_path):
    p = _new_provider(tmp_path)
    out = p.handle_tool_call("agentmemory_drift", {})
    state = json.loads(out)
    assert "alert" in state
    assert state["alert"] is False  # no samples yet


def test_session_switch_resets_prefetch_cache(tmp_path):
    p = _new_provider(tmp_path)
    p._prefetched["session-1"] = "stale"
    p.on_session_switch("session-2", reset=True)
    assert p._prefetched == {}
    assert p._session_id == "session-2"


def test_no_anthropic_key_still_imports(tmp_path):
    # The provider must be importable + instantiable without an Anthropic key
    p = mod.AgentMemoryProvider()
    assert p.is_available() is True
    assert p.name == "agentmemory"


def test_tool_schemas_shape(tmp_path):
    p = _new_provider(tmp_path)
    schemas = p.get_tool_schemas()
    names = sorted(s["name"] for s in schemas)
    assert names == ["agentmemory_drift", "agentmemory_forget", "agentmemory_recall"]
    for s in schemas:
        assert s["parameters"]["type"] == "object"


if __name__ == "__main__":
    import tempfile

    failures = 0
    for fn_name in [k for k in globals() if k.startswith("test_")]:
        fn = globals()[fn_name]
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
                print(f"PASS {fn_name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {fn_name}: {e}")
            except Exception as e:
                failures += 1
                print(f"FAIL {fn_name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
