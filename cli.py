#!/usr/bin/env python
"""Paper-RAG CLI — Local paper knowledge base with Zotero integration."""

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import load_config, Config
from src.indexer import Indexer
from src.rag_engine import RAGEngine

console = Console()


def _find_config() -> str:
    """Find config.yaml in current dir or project root."""
    candidates = [
        Path("config.yaml"),
        Path(__file__).parent / "config.yaml",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "config.yaml"


def _preflight(config: Config):
    """Validate config before running commands."""
    warnings = []

    if config.embedding.provider == "local":
        if not config.embedding.model_path or not Path(config.embedding.model_path).exists():
            warnings.append(
                f"Local embedding model not found: {config.embedding.model_path}. "
                "Set embedding.model_path in config.yaml."
            )
    elif not config.embedding.api_key:
        # Try env var directly
        key = os.environ.get("PAPERRAG_EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if key:
            config.embedding.api_key = key
        else:
            warnings.append(
                "Embedding API key not set. Use config.yaml (api_key) or "
                "set PAPERRAG_EMBED_API_KEY / OPENAI_API_KEY env var."
            )

    if not config.llm.api_key:
        key = os.environ.get("PAPERRAG_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if key:
            config.llm.api_key = key
        else:
            warnings.append(
                "LLM API key not set. Use config.yaml (api_key) or "
                "set PAPERRAG_LLM_API_KEY / OPENAI_API_KEY env var."
            )

    return warnings


@click.group()
@click.option("--config", "-c", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def cli(ctx, config_path):
    """Paper-RAG: Local paper knowledge base with Zotero integration."""
    ctx.ensure_object(dict)
    path = config_path or _find_config()

    if not Path(path).exists():
        console.print(f"[red]Config file not found: {path}[/red]")
        console.print("[dim]Create a config.yaml based on the template, or run:[/dim]")
        console.print("[dim]  paper-rag --config /path/to/config.yaml <command>[/dim]")
        sys.exit(1)

    config = load_config(path)
    ctx.obj["config"] = config


@cli.command()
@click.option("--source", "-s", multiple=True, help="Source: 'zotero', 'pdf_dir:<path>', or PDF file path")
@click.option("--clear", is_flag=True, help="Clear existing index before building")
@click.pass_context
def index(ctx, source, clear):
    """Build (or rebuild) the paper index from Zotero or PDF files."""
    config: Config = ctx.obj["config"]

    warnings = _preflight(config)
    if warnings:
        for w in warnings:
            console.print(f"[yellow]WARNING: {w}[/yellow]")
        console.print()

    sources = list(source) if source else None

    console.print(Panel.fit(
        "[bold]Building Paper Index[/bold]\n"
        f"Sources: {sources or 'config defaults'}\n"
        f"Vector store: {config.vector_store.path}",
        border_style="green",
    ))

    indexer = Indexer(config)
    stats = indexer.run(sources=sources, clear=clear)

    # Report
    console.print()
    table = Table(title="Indexing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Papers indexed", str(stats.papers))
    table.add_row("Chunks created", str(stats.chunks))
    table.add_row("Skipped (no PDF)", str(stats.skipped))
    table.add_row("Failures", str(len(stats.failures)))
    console.print(table)

    if stats.failures:
        console.print("\n[red]Failures:[/red]")
        for f in stats.failures:
            console.print(f"  [red]-[/red] {f}")


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", type=int, default=None, help="Number of chunks to retrieve")
@click.option("--sources/--no-sources", default=True, help="Show source papers")
@click.pass_context
def ask(ctx, query, top_k, sources):
    """Ask a question about your papers (single-shot RAG)."""
    config: Config = ctx.obj["config"]

    warnings = _preflight(config)
    engine = RAGEngine(config)

    with console.status("[bold yellow]Searching and generating...[/bold yellow]"):
        result = engine.query(query, top_k=top_k, show_sources=sources)

    console.print()
    console.print(result["answer"])

    if result.get("sources"):
        console.print("\n[dim]━━━ Sources ━━━[/dim]")
        seen = set()
        for s in result["sources"]:
            title = s["metadata"].get("title", "Unknown")
            if title not in seen:
                seen.add(title)
                authors = s["metadata"].get("authors", "")
                year = s["metadata"].get("year", "")
                console.print(
                    f"  [dim]• {title}"
                    + (f" — {authors}" if authors else "")
                    + (f" ({year})" if year else "")
                    + f"  [score: {s['score']}][/dim]"
                )


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", type=int, default=None, help="Number of results")
@click.option("--collection", "-c", default=None, help="Filter by Zotero collection")
@click.option("--by-paper", is_flag=True, help="Group results by paper")
@click.pass_context
def search(ctx, query, top_k, collection, by_paper):
    """Search the paper index (retrieval only, no LLM generation)."""
    config: Config = ctx.obj["config"]

    engine = RAGEngine(config)

    if by_paper:
        grouped = engine.retriever.search_by_paper(query, top_papers=top_k or 5)
        for i, (title, chunks) in enumerate(grouped.items()):
            meta = chunks[0]["metadata"]
            author = meta.get("authors", "")
            year = meta.get("year", "")
            console.print(f"\n[bold]{i+1}. {title}[/bold]")
            if author or year:
                console.print(f"   [dim]{author}  {year}[/dim]")
            for ch in chunks[:2]:
                console.print(f"   [dim]score: {ch['score']}[/dim]")
                console.print(f"   {ch['text'][:300]}...")
    else:
        hits = engine.retriever.search(query, top_k=top_k, filter_collection=collection)
        console.print(engine.retriever.format_hits(hits))


@cli.command()
@click.pass_context
def chat(ctx):
    """Start an interactive chat session with your paper library."""
    config: Config = ctx.obj["config"]

    warnings = _preflight(config)

    store_count = 0
    try:
        from src.vector_store import VectorStore
        vs = VectorStore(config.vector_store.path, config.vector_store.collection)
        store_count = vs.count()
    except Exception:
        pass

    if store_count == 0:
        console.print("[yellow]No indexed papers found. Run 'paper-rag index' first.[/yellow]")
        if not click.confirm("Continue anyway?", default=False):
            return

    if warnings:
        for w in warnings:
            console.print(f"[yellow]WARNING: {w}[/yellow]")

    engine = RAGEngine(config)
    engine.chat()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show index statistics."""
    config: Config = ctx.obj["config"]

    try:
        from src.vector_store import VectorStore
        vs = VectorStore(config.vector_store.path, config.vector_store.collection)
        s = vs.stats()
        console.print(f"Collection: {s['name']}")
        console.print(f"Chunks:     {s['count']}")
        console.print(f"Path:       {s['path']}")
    except Exception as e:
        console.print(f"[red]Failed to read stats: {e}[/red]")


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def clear(ctx, force):
    """Clear the vector store (delete all indexed papers)."""
    config: Config = ctx.obj["config"]

    if not force:
        if not click.confirm("This will delete all indexed papers. Continue?"):
            return

    from src.vector_store import VectorStore
    vs = VectorStore(config.vector_store.path, config.vector_store.collection)
    vs.clear()
    console.print("[green]Index cleared.[/green]")


@cli.command()
def config_path():
    """Print the path to the config file being used."""
    path = _find_config()
    console.print(f"Config: {path}")
    if Path(path).exists():
        config = load_config(path)
        console.print(f"Embedding API: {config.embedding.api_base}")
        console.print(f"Embedding model: {config.embedding.model}")
        console.print(f"LLM API: {config.llm.api_base}")
        console.print(f"LLM model: {config.llm.model}")
        console.print(f"Vector store: {config.vector_store.path}")


if __name__ == "__main__":
    cli()
