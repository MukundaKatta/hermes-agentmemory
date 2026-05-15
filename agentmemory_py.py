"""
Python sibling of agentmemory.

Same three pieces, same design rules, same public shape as the JS library
(@mukundakatta/agentmemory on npm). Used by the Streamlit demo so the live
URL can run on Streamlit Community Cloud or HuggingFace Spaces.

Design rules (unchanged from the JS version):
  1. No background work.
  2. Real deletes (no tombstones).
  3. Pull, never push.
  4. Show the trace.
  5. BYO LLM (function-injected).
  6. Zero non-stdlib runtime deps for the core (this file).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class EpisodicEvent:
    id: str
    session_id: str
    ts: int  # ms
    kind: str
    text: str
    meta: Optional[dict] = None
    embedding: Optional[list[float]] = None
    score: Optional[float] = None  # populated only by retrieve()


# --------------------------- EpisodicStore ---------------------------


class EpisodicStore:
    """In-memory episodic store. Append-only API + real deletes."""

    def __init__(self, embedder: Optional[Callable[[str], list[float]]] = None):
        self.embedder = embedder
        self._events: list[EpisodicEvent] = []
        self._next_id = 1

    def append(
        self,
        session_id: str,
        kind: str,
        text: str,
        ts: Optional[int] = None,
        meta: Optional[dict] = None,
    ) -> EpisodicEvent:
        eid = f"e_{self._next_id}"
        self._next_id += 1
        embedding = self.embedder(text) if self.embedder else None
        ev = EpisodicEvent(
            id=eid,
            session_id=session_id,
            ts=ts if ts is not None else int(time.time() * 1000),
            kind=kind,
            text=text,
            meta=meta,
            embedding=embedding,
        )
        self._events.append(ev)
        return ev

    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        since_ts: Optional[int] = None,
        until_ts: Optional[int] = None,
        kind: Optional[str] = None,
        top_k: int = 5,
    ) -> list[EpisodicEvent]:
        candidates = [
            e
            for e in self._events
            if (session_id is None or e.session_id == session_id)
            and (since_ts is None or e.ts >= since_ts)
            and (until_ts is None or e.ts <= until_ts)
            and (kind is None or e.kind == kind)
        ]
        if (
            self.embedder
            and candidates
            and candidates[0].embedding is not None
        ):
            qv = self.embedder(query)
            for e in candidates:
                e.score = _cosine(qv, e.embedding or [])
        else:
            qw = _tokenize(query)
            for e in candidates:
                e.score = _keyword_overlap(qw, _tokenize(e.text))
        candidates.sort(key=lambda e: e.score or 0, reverse=True)
        return candidates[:top_k]

    def get(self, eid: str) -> Optional[EpisodicEvent]:
        return next((e for e in self._events if e.id == eid), None)

    def list(self, **filters) -> list[EpisodicEvent]:
        return [
            e
            for e in self._events
            if all(
                {
                    "session_id": lambda v: e.session_id == v,
                    "since_ts": lambda v: e.ts >= v,
                    "until_ts": lambda v: e.ts <= v,
                    "kind": lambda v: e.kind == v,
                }[k](v)
                for k, v in filters.items()
                if v is not None
            )
        ]

    def delete_event(self, eid: str) -> bool:
        before = len(self._events)
        self._events = [e for e in self._events if e.id != eid]
        return len(self._events) < before

    def delete_session(self, session_id: str) -> int:
        before = len(self._events)
        self._events = [e for e in self._events if e.session_id != session_id]
        return before - len(self._events)

    def delete_older_than(self, older_than_ts: int) -> int:
        before = len(self._events)
        self._events = [e for e in self._events if e.ts >= older_than_ts]
        return before - len(self._events)

    @property
    def size(self) -> int:
        return len(self._events)

    def sessions(self) -> list[str]:
        return sorted({e.session_id for e in self._events})


# --------------------------- OnDemandSummarizer ---------------------------


_DEFAULT_SYSTEM = """You are summarizing past agent interactions for an upcoming session.
Goal: produce a short, faithful summary the agent can use as context.
Rules:
- Stay strictly within what is in the events. Do not invent facts.
- Prefer specifics (names, decisions, numbers) over general impressions.
- If events conflict, note the conflict instead of resolving it.
- Use plain words, no hedging. No bullet lists unless the events themselves are list-like.
- Never exceed the maxTokens budget."""


@dataclass
class SummaryResult:
    summary: str
    trace: dict  # event_ids, max_tokens, intent, prompt


class OnDemandSummarizer:
    """Pull-model context builder. Bring your own LLM."""

    def __init__(
        self,
        llm: Callable[[str], str],
        max_tokens: int = 300,
        system_prompt: Optional[str] = None,
    ):
        if not callable(llm):
            raise ValueError(
                "OnDemandSummarizer: `llm` must be callable (str -> str)."
            )
        self.llm = llm
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM

    def summarize(
        self, events: list[EpisodicEvent], intent: str
    ) -> SummaryResult:
        if not events:
            return SummaryResult(
                summary="",
                trace={
                    "event_ids": [],
                    "max_tokens": self.max_tokens,
                    "intent": intent,
                    "prompt": "",
                },
            )
        prompt = self._build_prompt(events, intent)
        summary = (self.llm(prompt) or "").strip()
        return SummaryResult(
            summary=summary,
            trace={
                "event_ids": [e.id for e in events],
                "max_tokens": self.max_tokens,
                "intent": intent,
                "prompt": prompt,
            },
        )

    def _build_prompt(self, events: list[EpisodicEvent], intent: str) -> str:
        block = "\n".join(
            f"[{i + 1}] ({e.kind} @ ts={e.ts}) {e.text}"
            for i, e in enumerate(events)
        )
        return "\n".join(
            [
                self.system_prompt,
                "",
                f"MAX_TOKENS: {self.max_tokens}",
                "",
                f"INTENT FOR UPCOMING SESSION: {intent}",
                "",
                "EVENTS (most-relevant first):",
                block,
                "",
                "SUMMARY:",
            ]
        )


# --------------------------- MemoryDriftWatcher ---------------------------


@dataclass
class _DriftSample:
    ts: int
    scores: list[float]


class MemoryDriftWatcher:
    """Rolling-window detector for retrieval-quality drops."""

    def __init__(self, window_size: int = 20, drop_threshold: float = 0.15):
        self.window_size = window_size
        self.drop_threshold = drop_threshold
        self._samples: list[_DriftSample] = []

    def record(self, ts: int, scores: list[float]) -> None:
        self._samples.append(_DriftSample(ts=ts, scores=scores))
        if len(self._samples) > self.window_size:
            self._samples.pop(0)

    def state(self) -> dict:
        n = len(self._samples)
        if n < 4:
            return {
                "samples": n,
                "mean_recent": 0,
                "mean_baseline": 0,
                "drop_fraction": 0,
                "alert": False,
                "reason": "not enough samples yet",
            }
        half = n // 2
        baseline = self._samples[:half]
        recent = self._samples[half:]
        mb = _mean([_mean(s.scores) for s in baseline])
        mr = _mean([_mean(s.scores) for s in recent])
        df = (mb - mr) / mb if mb > 0 else 0
        alert = df >= self.drop_threshold
        return {
            "samples": n,
            "mean_recent": mr,
            "mean_baseline": mb,
            "drop_fraction": df,
            "alert": alert,
            "reason": (
                f"mean retrieval score dropped {df * 100:.1f}% from baseline"
                if alert
                else "retrieval quality stable"
            ),
        }

    def reset(self) -> None:
        self._samples = []


# --------------------------- helpers ---------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"cosine: length mismatch ({len(a)} vs {len(b)})")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _tokenize(s: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9\s]", " ", s.lower()).split() if w]


def _keyword_overlap(qw: list[str], cw: list[str]) -> float:
    if not qw:
        return 0.0
    cs = set(cw)
    return sum(1 for w in qw if w in cs) / len(qw)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
