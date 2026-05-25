from pathlib import Path
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from .config import Config
from .zotero_reader import ZoteroReader, PaperMeta, scan_pdf_directory
from .pdf_parser import PDFParser, ParsedPaper
from .chunker import PaperChunker, Chunk
from .embedder import create_embedder, APIEmbedder, LocalEmbedder
from .vector_store import VectorStore

console = Console()


@dataclass
class IndexStats:
    papers: int
    chunks: int
    skipped: int
    failures: list[str]


class Indexer:
    """Orchestrate the full indexing pipeline: read → parse → chunk → embed → store."""

    def __init__(self, config: Config):
        self.config = config
        self.parser = PDFParser(max_pages=config.pdf.max_pages_per_paper)
        self.chunker = PaperChunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            keep_abstract_separate=config.chunking.keep_abstract_separate,
            keep_metadata=config.chunking.keep_metadata,
        )
        self.embedder = None  # lazy init
        self.vector_store = None  # lazy init

    def run(self, sources: list[str] | None = None, clear: bool = False) -> IndexStats:
        """Run the full indexing pipeline.

        Args:
            sources: "zotero", "pdf_dir:<path>", or list of PDF file paths.
                     None = use config defaults (try zotero first, then pdf_dir).
            clear: If True, clear the vector store before indexing.
        """
        # Init stores
        self._init_stores(clear)

        # Gather papers
        papers = self._gather_papers(sources)
        if not papers:
            console.print("[yellow]No papers found to index.[/yellow]")
            return IndexStats(papers=0, chunks=0, skipped=0, failures=[])

        console.print(f"[bold]Found {len(papers)} papers. Starting indexing...[/bold]\n")

        stats = IndexStats(papers=0, chunks=0, skipped=0, failures=[])

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing papers", total=len(papers))

            # Process in batches for embedding efficiency
            batch_chunks: list[Chunk] = []
            batch_size = self.config.embedding.batch_size

            for paper in papers:
                try:
                    chunks = self._process_paper(paper)
                    if not chunks:
                        stats.skipped += 1
                    else:
                        stats.papers += 1
                        for chunk in chunks:
                            batch_chunks.append(chunk)
                            if len(batch_chunks) >= batch_size:
                                self._embed_and_store(batch_chunks)
                                stats.chunks += len(batch_chunks)
                                batch_chunks = []

                except Exception as e:
                    title = paper.title or paper.item_key
                    stats.failures.append(f"{title}: {e}")
                    console.print(f"[red]  FAILED:[/red] {title} — {e}")

                progress.update(task, advance=1)

            # Remaining batch
            if batch_chunks:
                self._embed_and_store(batch_chunks)
                stats.chunks += len(batch_chunks)

        return stats

    def _process_paper(self, paper: PaperMeta) -> list[Chunk]:
        """Process one paper: parse PDF → chunk."""
        # Find PDF file
        pdf_path = self._find_pdf(paper)
        if not pdf_path:
            return []

        # Parse
        parsed = self.parser.parse(pdf_path)

        # Merge Zotero metadata into parsed result
        parsed.item_key = paper.item_key
        if paper.title and not parsed.title:
            parsed.title = paper.title
        if paper.authors and not parsed.authors:
            parsed.authors = paper.authors
        if paper.abstract and not parsed.abstract:
            parsed.abstract = paper.abstract

        # Chunk
        return self.chunker.chunk(paper, parsed)

    def _find_pdf(self, paper: PaperMeta) -> str | None:
        """Find a readable PDF for a paper."""
        for att in paper.attachments:
            if att.storage_path:
                p = Path(att.storage_path)
                if not p.exists():
                    continue
                if p.is_file():
                    return str(p)
                # May be a directory — look for any PDF inside
                for f in p.iterdir():
                    if f.suffix.lower() == ".pdf":
                        return str(f)

        return None

    def _gather_papers(self, sources: list[str] | None) -> list[PaperMeta]:
        """Collect papers from specified sources."""
        papers = []

        if sources:
            for src in sources:
                if src == "zotero":
                    papers.extend(self._read_zotero())
                elif src.startswith("pdf_dir:"):
                    dir_path = src.split(":", 1)[1]
                    papers.extend(scan_pdf_directory(dir_path))
                else:
                    # Assume it's a PDF file path
                    p = Path(src)
                    if p.suffix.lower() == ".pdf":
                        papers.append(PaperMeta(
                            item_key=str(p),
                            title=p.stem,
                            attachments=[type("Att", (), {
                                "item_key": str(p),
                                "filename": p.name,
                                "storage_path": str(p),
                            })()],
                        ))
        else:
            # Default: try Zotero first, then pdf_dir from config
            try:
                papers = self._read_zotero()
                if papers:
                    return papers
            except Exception:
                pass

            if self.config.pdf_dir.path:
                papers = scan_pdf_directory(
                    self.config.pdf_dir.path,
                    self.config.pdf_dir.recursive
                )

        return papers

    def _read_zotero(self) -> list[PaperMeta]:
        zotero_cfg = self.config.zotero
        reader = ZoteroReader(
            data_dir=zotero_cfg.data_dir,
            profile=zotero_cfg.profile,
        )
        console.print(f"[dim]Reading from Zotero: {reader.db_path}[/dim]")
        return reader.get_all_papers()

    def _init_stores(self, clear: bool):
        self.embedder = create_embedder(self.config.embedding)
        self.vector_store = VectorStore(
            persist_path=self.config.vector_store.path,
            collection_name=self.config.vector_store.collection,
        )
        if clear:
            self.vector_store.clear()
            console.print("[yellow]Cleared existing vector store.[/yellow]")

    def _embed_and_store(self, chunks: list[Chunk]):
        """Embed a batch of chunks and store them."""
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add(chunks, embeddings)
