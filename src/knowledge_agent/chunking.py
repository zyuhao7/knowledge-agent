from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;\n])")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    step = max(1, size - overlap)
    return [text[start : start + size] for start in range(0, len(text), step) if text[start:]]


def _split_oversized(block: str, size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in split_sentences(block):
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(sentence, size, overlap))
            continue
        if current and len(current) + len(sentence) > size:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, size: int = 420, overlap: int = 80) -> list[str]:
    """Pack paragraphs into chunks, splitting long paragraphs and repeating a tail as overlap."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(_split_oversized(block, size, overlap) if len(block) > size else [block])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n{piece}".strip()
        else:
            current = f"{current}\n{piece}".strip() if current else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks
