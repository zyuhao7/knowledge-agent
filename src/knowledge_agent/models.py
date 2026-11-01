from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    ordinal: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    vector_score: float
    keyword_score: float
    title: str


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: str
    meta: dict = field(default_factory=dict)


@dataclass
class TraceEvent:
    step: str
    detail: str


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    mode: str = "offline"
