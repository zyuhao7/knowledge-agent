# knowledge-agent

一个最小可运行的企业知识库 Agent:

> 熟悉 LLM 应用与 Agent 工作流落地：LLM、Agent、Prompt 工程、Function Calling、MCP、RAG

项目刻意保持小而完整：检索、工具调用、MCP 服务、评测四件事都能跑起来，每一层都可以单独替换成生产方案。

## 架构

```text
用户问题
   │
   ▼
Agent (agent.py)
   ├─ 离线模式：规则路由 → search_docs / read_doc / calculate
   └─ LLM 模式：Chat Completions + Function Calling 循环（最多 4 步）
          │
          ▼
ToolRegistry (tools.py)          MCP Server (mcp_server.py)
   ├─ search_docs  ───────────────┐
   ├─ read_doc                    │ 同一套工具
   ├─ list_documents              │ 通过 stdio 暴露给任意 MCP 客户端
   └─ calculate                   │
          │                       │
          ▼                       ▼
KnowledgeBase (rag.py) —— 分块、入库、混合检索
   ├─ chunking.py        按段落/句子切分，带 overlap
   ├─ embeddings.py      本地哈希向量（默认）/ OpenAI Embedding（可选）
   └─ store.py           cosine + BM25 混合打分
```

## 快速开始

```bash
cd knowledge-agent
uv sync
source .venv/bin/activate

# 或复用已有虚拟环境
# uv pip install --python /path/to/venv/bin/python -e . pytest ruff

# 完全离线，不需要 API Key
kagent docs
kagent ask "年假有几天？"
kagent search "出差住宿标准"
kagent tools

# 运行评测集
kagent eval --strict

# 单元测试
pytest
```

### 接入真实模型

复制 `.env.example` 为 `.env`，填入任意 OpenAI 兼容服务的地址和 Key：

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
TOP_K=5
```

之后加 `--llm` 即可走 Function Calling：

```bash
kagent ask "P0 故障多久响应？" --llm --trace
kagent eval --llm
```

没有 Key 时 Agent 自动使用确定性的离线路由，测试和评测不依赖网络。

### 远程 Embedding（可选）

默认使用本地 `HashingEmbedder`，零依赖、可复现。需要真实语义向量时：

```bash
USE_REMOTE_EMBEDDINGS=1 kagent search "报销多久到账"
```

## MCP 接入

任何支持 MCP 的客户端（Claude Desktop、IDE、Agent 框架）都可以直接使用这套工具：

```json
{
  "mcpServers": {
    "knowledge-agent": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "knowledge_agent", "mcp"],
      "cwd": "/path/to/knowledge-agent"
    }
  }
}
```

暴露的工具：`search_docs`、`read_doc`、`list_documents`、`calculate`。

## 评测

`evals/cases.jsonl` 每行一个用例，字段包括问题、期望文档、期望关键词、期望工具。`kagent eval` 会同时计算：

- **检索命中率**：期望文档是否出现在 Top-K
- **引用命中率**：答案是否引用了期望文档
- **关键词覆盖率**：答案是否包含关键事实
- **工具命中**：是否调用了期望工具
- **总体通过率**：以上条件同时满足

`--strict` 在任意用例失败时返回退出码 1，可直接接 CI。

## 代码地图

| 文件 | 作用 |
| --- | --- |
| `src/knowledge_agent/agent.py` | Agent 循环、Prompt、离线路由、引用抽取 |
| `src/knowledge_agent/tools.py` | 工具注册表、Function Calling schema、安全计算器 |
| `src/knowledge_agent/rag.py` | 文档加载、分块、检索入口 |
| `src/knowledge_agent/store.py` | 向量 + BM25 混合检索 |
| `src/knowledge_agent/embeddings.py` | 中文分词、哈希向量、远程 Embedding |
| `src/knowledge_agent/mcp_server.py` | MCP stdio 服务（MCP Python SDK v2） |
| `src/knowledge_agent/eval.py` | 评测执行与指标汇总 |
| `src/knowledge_agent/cli.py` | `kagent` 命令行 |
| `data/docs/*.md` | 示例内部制度文档 |
| `tests/` | 分块、检索、工具、Agent 的离线测试 |

## 示例问题

- 年假有几天？
- 出差住宿标准是多少？
- 报销多久到账？
- P0 故障多久响应？
- 请给出差旅住宿标准全文
- 128*3+16

## 可以继续扩展的方向

- 把 `HashingEmbedder` 换成 bge-m3 / text-embedding-3-small，并加入 rerank
- 文档侧接入 PDF、飞书、Notion 等真实数据源
- 用 LangGraph 或手写状态机替换当前的固定循环，加入规划与反思
- 增加多轮会话记忆、权限过滤、审计日志
- 把评测集扩大到线上真实问题，加入 LLM-as-judge
