import asyncio

from knowledge_agent.config import Settings
from knowledge_agent.mcp_server import build_server


def test_mcp_server_exposes_agent_tools() -> None:
    server = build_server(Settings.from_env())

    async def inspect():
        tools = await server.list_tools()
        result = await server.call_tool("calculate", {"expression": "128*3+16"})
        return tools, result

    tools, result = asyncio.run(inspect())
    assert {tool.name for tool in tools} == {
        "search_docs",
        "read_doc",
        "list_documents",
        "calculate",
    }
    assert "400" in str(result)
