from __future__ import annotations

import ast
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass

from .models import ToolResult
from .rag import KnowledgeBase

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> float:
    """Arithmetic only: no names, calls, attributes or comprehensions reach eval."""
    expression = expression.strip()
    if not expression or len(expression) > 200:
        raise ValueError("expression must be 1-200 characters")
    tree = ast.parse(expression, mode="eval")

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent too large")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    return walk(tree)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], ToolResult]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.specs: dict[str, ToolSpec] = {}
        self._register(
            ToolSpec(
                name="search_docs",
                description="Search the internal knowledge base and return ranked snippets with citations.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query.",
                        },
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                    "required": ["query"],
                },
                handler=self._search,
            )
        )
        self._register(
            ToolSpec(
                name="read_doc",
                description="Read a document's full text by its doc_id.",
                parameters={
                    "type": "object",
                    "properties": {"doc_id": {"type": "string"}},
                    "required": ["doc_id"],
                },
                handler=self._read,
            )
        )
        self._register(
            ToolSpec(
                name="list_documents",
                description="List every document in the knowledge base.",
                parameters={"type": "object", "properties": {}},
                handler=self._list,
            )
        )
        self._register(
            ToolSpec(
                name="calculate",
                description="Evaluate a basic arithmetic expression such as 128*3+16.",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                handler=self._calculate,
            )
        )

    def _register(self, spec: ToolSpec) -> None:
        self.specs[spec.name] = spec

    def schemas(self) -> list[dict]:
        return [spec.schema() for spec in self.specs.values()]

    def dispatch(self, name: str, arguments: dict) -> ToolResult:
        spec = self.specs.get(name)
        if not spec:
            return ToolResult(name=name, ok=False, content=f"unknown tool: {name}")
        try:
            return spec.handler(arguments)
        except Exception as error:  # noqa: BLE001 - tool errors go back to the model
            return ToolResult(name=name, ok=False, content=f"{type(error).__name__}: {error}")

    def _search(self, arguments: dict) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        top_k = int(arguments.get("top_k") or 5)
        top_k = max(1, min(10, top_k))
        hits = self.kb.search(query, top_k=top_k)
        payload = [
            {
                "citation": hit.chunk.id,
                "doc_id": hit.chunk.doc_id,
                "title": hit.title,
                "score": round(hit.score, 4),
                "snippet": hit.chunk.text[:500],
            }
            for hit in hits
        ]
        return ToolResult(
            name="search_docs",
            ok=True,
            content=json.dumps(payload, ensure_ascii=False),
            meta={"doc_ids": sorted({hit.chunk.doc_id for hit in hits})},
        )

    def _read(self, arguments: dict) -> ToolResult:
        doc_id = str(arguments.get("doc_id") or "").strip()
        document = self.kb.read_doc(doc_id)
        if not document:
            return ToolResult(name="read_doc", ok=False, content=f"unknown doc_id: {doc_id}")
        return ToolResult(
            name="read_doc",
            ok=True,
            content=document.text[:6000],
            meta={"doc_ids": [document.id], "title": document.title},
        )

    def _list(self, arguments: dict) -> ToolResult:
        payload = self.kb.catalog()
        return ToolResult(
            name="list_documents",
            ok=True,
            content=json.dumps(payload, ensure_ascii=False),
            meta={"doc_ids": [item["doc_id"] for item in payload]},
        )

    def _calculate(self, arguments: dict) -> ToolResult:
        expression = str(arguments.get("expression") or "")
        value = safe_calculate(expression)
        rendered = int(value) if value.is_integer() else round(value, 8)
        return ToolResult(name="calculate", ok=True, content=f"{expression} = {rendered}")
