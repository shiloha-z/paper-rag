# Paper-RAG

本地论文知识库，连接 Zotero 或本地 PDF 目录，构建向量索引，支持语义检索和 RAG 问答。

## 安装

```bash
pip install -r requirements.txt
```

## 配置

编辑 `config.yaml`，填入 API 信息：

```yaml
embedding:
  api_base: "https://your-api.com/v1"
  api_key: "${OPENAI_API_KEY}"     # 支持 ${ENV_VAR} 或直接写 key
  model: "text-embedding-3-small"

llm:
  api_base: "https://your-api.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o"
```

也可以通过环境变量设置 key：

- `PAPERRAG_EMBED_API_KEY` 或 `OPENAI_API_KEY` → Embedding
- `PAPERRAG_LLM_API_KEY` 或 `OPENAI_API_KEY` → LLM

Embedding 和 LLM 可以指向不同的 API 和模型，各自独立配置。

## 命令

### 建索引

```bash
# 从 Zotero 读取（自动检测安装路径）
python cli.py index -s zotero

# 从本地 PDF 目录扫描
python cli.py index -s pdf_dir:f:/papers

# 混合来源
python cli.py index -s zotero -s pdf_dir:f:/more_papers

# 清空后重建
python cli.py index -s zotero --clear
```

### 语义检索（仅检索，不含 LLM）

```bash
# 基础搜索
python cli.py search "transformer attention mechanism"

# 按论文聚合结果
python cli.py search "graph neural network" --by-paper

# 指定返回数量 & Zotero collection 过滤
python cli.py search "contrastive learning" -k 10 -c "NLP"
```

### RAG 问答（检索 + LLM 生成）

```bash
# 单次问答
python cli.py ask "attention 机制和 CNN 的本质区别是什么"

# 不显示来源
python cli.py ask "summarize the key findings" --no-sources
```

### 交互式对话

```bash
python cli.py chat
```

对话中可用指令：

| 指令 | 作用 |
|------|------|
| `/search <query>` | 纯检索，不走 LLM |
| `/papers <query>` | 按论文聚合检索结果 |
| `/quit` | 退出 |

### 管理

```bash
python cli.py stats          # 查看索引统计
python cli.py clear          # 清空向量库
python cli.py config-path    # 查看当前使用的配置
```

## MCP Server — 让外部模型调用知识库

启动 MCP server 后，任何支持 MCP 的客户端（Claude Desktop、VS Code、Cursor 等）都可以直接搜索和问答你的论文库。

### 在 Claude Desktop 中配置

编辑 Claude Desktop 配置文件（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "paper-rag": {
      "command": "python",
      "args": ["f:/REPO/RAG/mcp_server.py"],
      "cwd": "f:/REPO/RAG"
    }
  }
}
```

macOS / Linux 把路径换成你的实际路径。重启 Claude Desktop 后，对话中直接说"搜索我的论文库中关于 attention 的论文"即可自动调用。

### 在 VS Code 中配置

在 `settings.json` 中：

```json
"mcp.servers": {
  "paper-rag": {
    "command": "python",
    "args": ["f:/REPO/RAG/mcp_server.py"]
  }
}
```

### 提供的工具

MCP server 向模型暴露 4 个工具：

| 工具 | 用途 | 触发场景 |
|------|------|----------|
| `paper_search` | 语义搜索，返回 chunk 列表 | "我的论文库中有没有关于 X 的文献" |
| `paper_ask` | RAG 问答，搜索 + LLM 生成答案 | "对比我的论文库中 A 和 B 方法的区别" |
| `paper_structure_search` | 按论文聚合搜索，用于概览 | "梳理一下我这个领域的主要方向" |
| `paper_stats` | 索引统计 | "当前索引了多少篇论文" |

### 同时使用 CLI 和 MCP

CLI 在终端用，MCP 供编辑器/模型调用，它们共享同一份 `config.yaml` 和同一个 ChromaDB 索引，互不冲突。

## 从 Zotero 连接

系统直接读取 Zotero 本地的 `zotero.sqlite` 数据库，不需要：

- Zotero 处于运行状态
- Zotero API key
- WebDAV 同步

支持 Zotero 的 `journalArticle`、`conferencePaper`、`preprint`、`thesis`、`bookSection` 等条目类型，自动提取标题、作者、摘要、年份、标签、合集、DOI 等元数据。

如果 Zotero 安装在非标准路径，在 `config.yaml` 中指定：

```yaml
zotero:
  data_dir: "D:/Zotero"     # Zotero 数据目录
  profile: "xxxxxxxx.default"  # 可选，指定 profile 子目录
```

## 分块策略

每篇论文入库时按结构化方式切分：

- **标题 chunk**：论文标题 + 作者 + 年份，便于精确匹配
- **摘要 chunk**：独立入库，回答"有哪些相关论文"时命中率更高
- **正文 chunks**：按章节（Introduction / Method / Experiments / Conclusion）切分，带重叠保持上下文

所有 chunk 携带完整元数据（标题、作者、年份、期刊、DOI、标签、合集），检索结果可追溯。
