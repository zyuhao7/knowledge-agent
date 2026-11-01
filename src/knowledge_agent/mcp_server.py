from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .config import Settings
from .embeddings import get_embedder
from .rag import KnowledgeBase
from .tools import ToolRegistry


def build_server(settings: Settings) -> MCPServer:
    kb = KnowledgeBase(
        settings.data_dir,
        get_embedder(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    kb.ingest()
    tools = ToolRegistry(kb)
    server = MCPServer("knowledge-agent")

    @server.tool()
    def search_docs(query: str, top_k: int = 5) -> str:
        """Search the internal knowledge base. Returns JSON hits with citation ids and snippets."""
        return tools.dispatch("search_docs", {"query": query, "top_k": top_k}).content

    @server.tool()
    def read_doc(doc_id: str) -> str:
        """Read a document's full text by doc_id, for example leave-policy."""
        return tools.dispatch("read_doc", {"doc_id": doc_id}).content

    @server.tool()
    def list_documents() -> str:
        """List every document in the knowledge base."""
        return tools.dispatch("list_documents", {}).content

    @server.tool()
    def calculate(expression: str) -> str:
        """Evaluate basic arithmetic such as 128*3+16."""
        return tools.dispatch("calculate", {"expression": expression}).content

    return server


def serve(settings: Settings) -> None:
    build_server(settings).run()
