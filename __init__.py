"""agentmemory plugin for Hermes Agent.

Pull-model episodic memory with real deletes and audit trace.

Drop into Hermes Agent's plugins/memory/agentmemory/ directory, then activate via:
    hermes config set memory.provider agentmemory

Design rules (inherited from agentmemory):
  1. No background work. Memory does not consolidate while the agent is idle.
  2. Real deletes. When a session is deleted, every byte is gone.
  3. Pull, never push. The agent retrieves memory explicitly.
  4. Show the trace. Every prefetch logs the event ids and the prompt.
  5. BYO LLM. The summarizer reuses the agent's configured Claude key.

Configuration (env vars):
  AGENTMEMORY_TOP_K          number of past events to retrieve (default 5)
  AGENTMEMORY_MAX_TOKENS     summary token budget per prefetch (default 300)
  AGENTMEMORY_MODEL          Claude model id (default claude-sonnet-4-5-20251022)
  AGENTMEMORY_TRACE_LOG      file path to append prefetch trace JSON (default $HERMES_HOME/agentmemory/trace.jsonl)
  ANTHROPIC_API_KEY          Anthropic key for summarizer (required)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .agentmemory_py import (
    EpisodicStore,
    MemoryDriftWatcher,
    OnDemandSummarizer,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5-20251022"


def _make_anthropic_llm(model: str, max_tokens: int):
    """Return a callable str -> str that calls Anthropic with the user's key.

    Imported lazily so the plugin still imports cleanly when anthropic is not
    installed (the call site raises a clear error instead).
    """

    def call(prompt: str) -> str:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "agentmemory plugin: anthropic package not installed. "
                "Run `pip install anthropic`."
            ) from e
        client = Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # join all text blocks
        return "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )

    return call


class AgentMemoryProvider(MemoryProvider):
    """Pull-model episodic memory provider for Hermes Agent."""

    def __init__(self):
        self._store = EpisodicStore()
        self._summarizer: Optional[OnDemandSummarizer] = None
        self._watcher = MemoryDriftWatcher()
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._top_k = int(os.getenv("AGENTMEMORY_TOP_K", "5"))
        self._max_tokens = int(os.getenv("AGENTMEMORY_MAX_TOKENS", "300"))
        self._model = os.getenv("AGENTMEMORY_MODEL", DEFAULT_MODEL)
        self._trace_path: Optional[Path] = None
        self._trace_lock = threading.Lock()
        self._prefetched: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "agentmemory"

    # -- Core lifecycle ------------------------------------------------------

    def is_available(self) -> bool:
        # Always importable; Anthropic key is only needed at summarize-time so
        # we report available even without it. The summarizer itself raises a
        # clear error on first call if the key is missing.
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home") or os.path.expanduser("~/.hermes")
        trace_dir = Path(self._hermes_home) / "agentmemory"
        trace_dir.mkdir(parents=True, exist_ok=True)
        env_path = os.getenv("AGENTMEMORY_TRACE_LOG")
        self._trace_path = Path(env_path) if env_path else trace_dir / "trace.jsonl"
        self._summarizer = OnDemandSummarizer(
            llm=_make_anthropic_llm(self._model, self._max_tokens),
            max_tokens=self._max_tokens,
        )
        logger.info(
            "agentmemory initialized: session=%s home=%s trace=%s top_k=%d",
            session_id,
            self._hermes_home,
            self._trace_path,
            self._top_k,
        )

    def system_prompt_block(self) -> str:
        return (
            "You have access to agentmemory: pull-model episodic memory.\n"
            "- past events from earlier sessions can be summarized into context on demand;\n"
            "- the user can audit and delete every event;\n"
            "- nothing is silently injected into your prompt without a trace entry."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        sid = session_id or self._session_id
        cached = self._prefetched.pop(sid, "")
        if cached:
            return cached
        return self._do_prefetch(query, sid)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Synchronous on purpose. Background work is the design rule we exist
        # to push against; if the user wants async they can override.
        sid = session_id or self._session_id
        self._prefetched[sid] = self._do_prefetch(query, sid)

    def _do_prefetch(self, query: str, session_id: str) -> str:
        if not self._summarizer:
            return ""
        events = self._store.retrieve(query, top_k=self._top_k)
        if not events:
            self._watcher.record(int(__import__("time").time() * 1000), [])
            return ""
        scores = [e.score or 0.0 for e in events]
        self._watcher.record(int(__import__("time").time() * 1000), scores)
        result = self._summarizer.summarize(events, intent=query)
        self._write_trace(
            {
                "session_id": session_id,
                "intent": query,
                "event_ids": result.trace["event_ids"],
                "summary": result.summary,
                "drift": self._watcher.state(),
            }
        )
        return result.summary

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        sid = session_id or self._session_id
        if user_content:
            self._store.append(sid, "user", user_content)
        if assistant_content:
            self._store.append(sid, "assistant", assistant_content)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "agentmemory_recall",
                "description": (
                    "Search the user's past conversations for relevant context. "
                    "Returns a short summary of the top matching events plus their event ids "
                    "so the user can audit which memories were used."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language description of what to recall.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of events to consider.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "agentmemory_forget",
                "description": (
                    "Delete past events. Use when the user asks to forget something. "
                    "Deletes are real: no tombstone is left behind."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Delete every event from this session.",
                        },
                        "event_id": {
                            "type": "string",
                            "description": "Delete a single event by id.",
                        },
                    },
                },
            },
            {
                "name": "agentmemory_drift",
                "description": (
                    "Report the rolling-window retrieval drift state. Helpful "
                    "when memory recall starts feeling stale."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "agentmemory_recall":
            query = args.get("query", "")
            top_k = int(args.get("top_k") or self._top_k)
            events = self._store.retrieve(query, top_k=top_k)
            if not self._summarizer or not events:
                return json.dumps({"summary": "", "event_ids": []})
            r = self._summarizer.summarize(events, intent=query)
            return json.dumps({"summary": r.summary, "event_ids": r.trace["event_ids"]})
        if tool_name == "agentmemory_forget":
            sid = args.get("session_id")
            eid = args.get("event_id")
            removed = 0
            if sid:
                removed += self._store.delete_session(sid)
            if eid:
                removed += 1 if self._store.delete_event(eid) else 0
            return json.dumps({"removed": removed})
        if tool_name == "agentmemory_drift":
            return json.dumps(self._watcher.state())
        raise NotImplementedError(f"unknown tool: {tool_name}")

    # -- Optional hooks ------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # No background extraction. The Build value of agentmemory is that
        # session end is uneventful: events are already on disk, and the
        # summary will be rebuilt on demand at the start of the next session.
        return

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        if reset:
            self._prefetched.clear()
        self._session_id = new_session_id

    def shutdown(self) -> None:
        # Nothing to flush — every write is synchronous.
        return

    # -- Helpers ------------------------------------------------------------

    def _write_trace(self, record: Dict[str, Any]) -> None:
        if not self._trace_path:
            return
        try:
            with self._trace_lock:
                with open(self._trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        except OSError as e:  # pragma: no cover
            logger.warning("agentmemory: failed to write trace: %s", e)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "anthropic_api_key",
                "description": "Anthropic API key for the summarizer.",
                "secret": True,
                "required": True,
                "env_var": "ANTHROPIC_API_KEY",
                "url": "https://console.anthropic.com/settings/keys",
            },
            {
                "key": "model",
                "description": "Claude model id used for summarization.",
                "required": False,
                "default": DEFAULT_MODEL,
                "env_var": "AGENTMEMORY_MODEL",
            },
            {
                "key": "top_k",
                "description": "How many past events to consider per prefetch.",
                "required": False,
                "default": "5",
                "env_var": "AGENTMEMORY_TOP_K",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        # All non-secret config is read from env vars at init time, so we
        # write a tiny JSON next to the trace log for `hermes memory status`
        # to surface what's active.
        cfg_dir = Path(hermes_home) / "agentmemory"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2)


# Hermes' plugin loader looks for a top-level `provider` factory by default.
def provider() -> AgentMemoryProvider:
    return AgentMemoryProvider()
