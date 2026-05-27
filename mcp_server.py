#!/usr/bin/env python
"""MCP server for Paper-RAG — exposes paper search and RAG tools to LLM clients.

Usage (via .mcp.json in project root):
    Claude Code auto-detects .mcp.json and spawns this server on demand.

    The server starts fast (< 1s) — BGE-M3 model is lazy-loaded on first tool call.
"""

import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.config import load_config, Config
from src.rag_engine import RAGEngine

config: Config = None
_engine: RAGEngine = None

server = Server("paper-rag")


def _get_engine() -> RAGEngine:
    """Lazy-load the RAG engine so MCP init doesn't timeout."""
    global _engine
    if _engine is None:
        _engine = RAGEngine(config)
    return _engine

# --- Tool implementations ---


def _format_hits(hits: list[dict], max_per_hit: int = 400) -> str:
    if not hits:
        return "No matching papers found."

    lines = []
    for i, h in enumerate(hits):
        m = h.get("metadata", {})
        header = f"[{i + 1}] {m.get('title', 'Unknown')}"
        if m.get("authors"):
            header += f" — {m['authors']}"
        if m.get("year"):
            header += f" ({m['year']})"
        if m.get("journal"):
            header += f" | {m['journal']}"
        ctype = m.get("chunk_type", "")
        if ctype:
            header += f" [{ctype}]"
        lines.append(f"{header}\nScore: {h['score']}\n{h['text'][:max_per_hit]}")
    return "\n\n---\n\n".join(lines)


def _tool_search(arguments: dict) -> str:
    query = arguments.get("query", "")
    top_k = min(arguments.get("top_k", 8), 20)
    collection = arguments.get("collection", "")
    if not query:
        return "Error: query is required."
    hits = _get_engine().retriever.search(query, top_k=top_k, filter_collection=collection or None)
    return _format_hits(hits)


def _tool_ask(arguments: dict) -> str:
    question = arguments.get("question", "")
    top_k = min(arguments.get("top_k", 8), 20)
    if not question:
        return "Error: question is required."
    result = _get_engine().query(question, top_k=top_k, show_sources=False)
    return result["answer"]


def _tool_structure_search(arguments: dict) -> str:
    query = arguments.get("query", "")
    top_k = min(arguments.get("top_k", 12), 30)
    if not query:
        return "Error: query is required."
    grouped = _get_engine().retriever.search_by_paper(query, top_papers=top_k)

    if not grouped:
        return "No matching papers found."

    lines = []
    for i, (title, chunks) in enumerate(grouped.items()):
        m = chunks[0]["metadata"]
        header = f"[{i + 1}] {title}"
        if m.get("authors"):
            header += f" — {m['authors']}"
        if m.get("year"):
            header += f" ({m['year']})"
        best_score = max(ch["score"] for ch in chunks)
        lines.append(f"{header}  [relevance: {best_score:.3f}]")
        text_preview = chunks[0]["text"][:250].replace("\n", " ")
        lines.append(f"   {text_preview}...")
    return "\n\n".join(lines)


def _tool_stats(_arguments: dict) -> str:
    stats = _get_engine().store.stats()
    return json.dumps(stats, indent=2, ensure_ascii=False)


TOOLS = {
    "paper_search": (
        "Semantic search across the paper knowledge base. "
        "Use when finding papers by topic, methodology, or research question. "
        "Returns ranked chunks with metadata (title, authors, year, journal, score).",
        {
            "query": ("string", "Natural language search query."),
            "top_k": ("integer", "Number of results (default 8, max 20)."),
            "collection": ("string", "Optional Zotero collection name filter."),
        },
        _tool_search,
    ),
    "paper_ask": (
        "RAG-powered question answering over the paper knowledge base. "
        "Use when asking questions that require synthesizing findings, "
        "comparing methods, or generating citations for writing tasks.",
        {
            "question": ("string", "A question about the papers. Be specific."),
            "top_k": ("integer", "Number of chunks for context (default 8, max 20)."),
        },
        _tool_ask,
    ),
    "paper_structure_search": (
        "Search papers grouped by title, best for surveying a topic. "
        "Returns paper-level overview rather than individual chunks.",
        {
            "query": ("string", "Natural language search query."),
            "top_k": ("integer", "Max papers to report (default 12, max 30)."),
        },
        _tool_structure_search,
    ),
    "paper_stats": (
        "Get paper index statistics: total chunks, collection name, storage path.",
        {},
        _tool_stats,
    ),
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available tools."""
    tools = []
    for name, (description, params, _) in TOOLS.items():
        properties = {}
        required = []
        for pname, (ptype, pdesc) in params.items():
            properties[pname] = {"type": ptype, "description": pdesc}
            # query/question are required
            if pname in ("query", "question"):
                required.append(pname)

        input_schema = {"type": "object", "properties": properties}
        if required:
            input_schema["required"] = required

        tools.append(Tool(name=name, description=description, inputSchema=input_schema))
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle a tool call."""
    if name not in TOOLS:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        handler = TOOLS[name][2]
        result = handler(arguments)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Tool error: {e}")]


# --- Startup ---

def _find_config(config_path: str = "") -> str:
    if config_path:
        p = Path(config_path)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Config not found: {config_path}")

    for c in [Path("config.yaml"), Path(__file__).parent / "config.yaml"]:
        if c.exists():
            return str(c)
    raise FileNotFoundError("config.yaml not found")


async def main():
    global config, engine

    args = [a for a in sys.argv[1:] if not a.startswith("--mcp")]
    config_path = ""
    for i, a in enumerate(args):
        if a == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            break

    config_path = _find_config(config_path)
    config = load_config(config_path)
    # Engine is lazy-loaded on first tool call to avoid MCP init timeout

    import logging
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] paper-rag: %(message)s",
    )
    logger = logging.getLogger("paper-rag")
    logger.info("MCP server starting (engine lazy-loaded)")

    try:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    except Exception:
        logger.exception("Server crashed")
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
