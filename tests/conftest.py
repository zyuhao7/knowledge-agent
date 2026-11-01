from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_agent.config import ROOT, Settings
from knowledge_agent.embeddings import HashingEmbedder
from knowledge_agent.rag import KnowledgeBase
from knowledge_agent.tools import ToolRegistry


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    settings = Settings.from_env(ROOT / "data" / "docs")
    base = KnowledgeBase(
        settings.data_dir,
        HashingEmbedder(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    base.ingest()
    return base


@pytest.fixture(scope="session")
def tools(kb: KnowledgeBase) -> ToolRegistry:
    return ToolRegistry(kb)


def test_documents_are_ingested(kb: KnowledgeBase) -> None:
    assert len(kb.catalog()) >= 4
    assert kb.store.chunks
    assert all(Path(document.source).exists() for document in kb.documents.values())
