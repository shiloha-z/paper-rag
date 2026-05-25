# Paper-RAG

本地论文知识库，连接 Zotero 或本地 PDF 目录，构建向量索引，支持语义检索和 RAG 问答。

## 安装

```bash
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果使用 GPU（推荐），需安装 CUDA 版 torch：

```bash
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu132
```

## 配置

### 1. 创建配置文件

```bash
cp config.example.yaml config.yaml
```

### 2. 创建 `.env` 放 API key

```env
DEEPSEEK_API_KEY=sk-your-key
MODEL_NAME=deepseek-v4-flash
API_BASE=https://api.deepseek.com/v1
```

`config.yaml` 用 `${VAR}` 语法引用 `.env` 中的变量，系统会自动加载。

### 3. Embedding 选择

**本地模型（推荐）** — BGE-M3，中英文都好，离线可用：

```yaml
embedding:
  provider: "local"
  model_path: "F:/ComfyUI/models/embeddings/bge-m3"
  batch_size: 64
  device: "cuda"
```

**API 模型** — OpenAI 兼容接口：

```yaml
embedding:
  provider: "api"
  api_base: "${API_BASE}"
  api_key: "${DEEPSEEK_API_KEY}"
  model: "text-embedding-3-small"
```

### 4. LLM

```yaml
llm:
  api_base: "${API_BASE}"
  api_key: "${DEEPSEEK_API_KEY}"
  model: "${MODEL_NAME}"
```

Embedding 和 LLM 可以指向不同的 API，各自独立配置。

## 命令

所有命令在项目目录下用虚拟环境运行。

### 建索引

```bash
# 从 Zotero 读取（自动检测安装路径）
.\venv\Scripts\python.exe cli.py index -s zotero

# 从本地 PDF 目录扫描
.\venv\Scripts\python.exe cli.py index -s pdf_dir:f:/papers

# 清空后重建（解决重复索引）
.\venv\Scripts\python.exe cli.py index -s zotero --clear
```

### 语义检索（仅检索，不含 LLM）

```bash
.\venv\Scripts\python.exe cli.py search "transformer attention mechanism"
.\venv\Scripts\python.exe cli.py search "graph neural network" --by-paper
.\venv\Scripts\python.exe cli.py search "contrastive learning" -k 10 -c "NLP"
```

### RAG 问答（检索 + LLM 生成）

```bash
.\venv\Scripts\python.exe cli.py ask "attention 机制和 CNN 的本质区别是什么"
.\venv\Scripts\python.exe cli.py ask "summarize the key findings" --no-sources
```

### 交互式对话

```bash
.\venv\Scripts\python.exe cli.py chat
```

对话中可用指令：

| 指令 | 作用 |
|------|------|
| `/search <query>` | 纯检索，不走 LLM |
| `/papers <query>` | 按论文聚合检索结果 |
| `/quit` | 退出 |

### 管理

```bash
.\venv\Scripts\python.exe cli.py stats          # 查看索引统计
.\venv\Scripts\python.exe cli.py clear          # 清空向量库
```

## MCP Server — 让外部模型调用知识库

启动 MCP server 后，Claude Code 可以直接搜索和问答你的论文库。

### 配置

项目根目录已有 [`.mcp.json`](.mcp.json)，Claude Code 启动时自动识别。第一次可能弹窗问是否信任该 MCP server，确认即可。

如果路径不同，编辑 `.mcp.json` 修改 `command` 和 `args`：

```json
{
  "mcpServers": {
    "paper-rag": {
      "command": "F:/REPO/RAG/venv/Scripts/python.exe",
      "args": ["F:/REPO/RAG/mcp_server.py"],
      "cwd": "F:/REPO/RAG"
    }
  }
}
```

### 提供的工具

| 工具 | 用途 | 触发场景 |
|------|------|----------|
| `paper_search` | 语义搜索，返回 chunk 列表 | "我的论文库中有没有关于 X 的文献" |
| `paper_ask` | RAG 问答，搜索 + LLM 生成答案 | "对比我的论文库中 A 和 B 方法的区别" |
| `paper_structure_search` | 按论文聚合搜索，用于概览 | "梳理一下我这个领域的主要方向" |
| `paper_stats` | 索引统计 | "当前索引了多少篇论文" |

CLI 和 MCP 共享同一份配置和向量库，互不冲突。

## 从 Zotero 连接

系统直接读取 Zotero 本地的 `zotero.sqlite` 数据库，不需要 Zotero 运行、不需要 API key、不需要 WebDAV。

读取时会自动复制 DB 到临时目录，避免和正在运行的 Zotero 产生锁冲突。

支持 `journalArticle`、`conferencePaper`、`preprint`、`thesis`、`bookSection` 等条目类型，自动提取标题、作者、摘要、年份、标签、合集、DOI 等元数据。

如果 Zotero 安装在非标准路径，在 `config.yaml` 中指定：

```yaml
zotero:
  data_dir: "D:/Zotero"
  profile: "xxxxxxxx.default"
```

## 分块策略

每篇论文入库时按结构化方式切分：

- **标题 chunk**：论文标题 + 作者 + 年份，便于精确匹配
- **摘要 chunk**：独立入库，回答"有哪些相关论文"时命中率更高
- **正文 chunks**：按章节（Introduction / Method / Experiments / Conclusion）切分，带重叠保持上下文

所有 chunk 携带完整元数据（标题、作者、年份、期刊、DOI、标签、合集），检索结果可追溯。
