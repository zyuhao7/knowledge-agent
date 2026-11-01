from __future__ import annotations

from pathlib import Path

from .chunking import chunk_text
from .embeddings import Embedder
from .models import Chunk, Document, SearchHit
from .store import VectorStore


def _read_markdown(path: Path) -> Document:
    text = path.read_text(encoding="utf-8").strip()
    title = path.stem
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return Document(id=path.stem, title=title, text=text, source=str(path))


class KnowledgeBase:
    def __init__(
        self,
        data_dir: Path,
        embedder: Embedder,
        chunk_size: int = 420,
        chunk_overlap: int = 80,
    ) -> None:
        self.data_dir = data_dir
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: dict[str, Document] = {}
        self.store = VectorStore(embedder)

    def ingest(self) -> int:
        paths = sorted(self.data_dir.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"no markdown documents under {self.data_dir}")
        self.documents = {path.stem: _read_markdown(path) for path in paths}

        chunks: list[Chunk] = []
        for document in self.documents.values():
            for ordinal, text in enumerate(
                chunk_text(document.text, self.chunk_size, self.chunk_overlap)
            ):
                chunks.append(
                    Chunk(
                        id=f"{document.id}#{ordinal}",
                        doc_id=document.id,
                        ordinal=ordinal,
                        text=text,
                    )
                )
        self.store.add(chunks, {document.id: document.title for document in self.documents.values()})
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        return self.store.search(query, top_k=top_k)

    def read_doc(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def catalog(self) -> list[dict[str, str]]:
        return [
            {"doc_id": document.id, "title": document.title}
            for document in sorted(self.documents.values(), key=lambda item: item.id)
        ]

    @staticmethod
    def citation(hit: SearchHit) -> str:
        return hit.chunk.id
