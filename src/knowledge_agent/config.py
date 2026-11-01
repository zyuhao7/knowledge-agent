from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> None:
    """Read a simple KEY=VALUE .env without adding a dependency."""
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    chat_model: str
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    data_dir: Path

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> Settings:
        load_env()
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY") or None,
            base_url=(os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
            chat_model=os.environ.get("CHAT_MODEL") or "gpt-4.1-mini",
            embedding_model=os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small",
            top_k=int(os.environ.get("TOP_K") or 5),
            chunk_size=int(os.environ.get("CHUNK_SIZE") or 420),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP") or 80),
            data_dir=(data_dir or ROOT / "data" / "docs").resolve(),
        )
