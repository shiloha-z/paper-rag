from dataclasses import dataclass, field

from .pdf_parser import ParsedPaper
from .zotero_reader import PaperMeta


@dataclass
class Chunk:
    """A single text chunk from a paper, ready for embedding."""
    text: str
    metadata: dict = field(default_factory=dict)


class PaperChunker:
    """Split academic papers into chunks with metadata preservation."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        keep_abstract_separate: bool = True,
        keep_metadata: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_abstract_separate = keep_abstract_separate
        self.keep_metadata = keep_metadata

    def chunk(self, meta: PaperMeta, parsed: ParsedPaper) -> list[Chunk]:
        """Split a paper into chunks with rich metadata."""
        chunks = []

        base_meta = {}
        if self.keep_metadata:
            base_meta = {
                "item_key": meta.item_key or parsed.item_key,
                "title": meta.title or parsed.title,
                "authors": ", ".join(meta.authors) if meta.authors else "",
                "year": meta.year,
                "journal": meta.journal,
                "doi": meta.doi,
                "tags": ", ".join(meta.tags),
                "collections": ", ".join(meta.collections),
            }

        content_source = parsed.full_text if parsed.full_text else ""
        if not content_source.strip():
            return chunks

        # Abstract as a standalone chunk (high-value for retrieval)
        abstract = parsed.abstract or meta.abstract
        if self.keep_abstract_separate and abstract and len(abstract) > 50:
            chunks.append(Chunk(
                text=f"ABSTRACT: {abstract.strip()}",
                metadata={**base_meta, "chunk_type": "abstract"},
            ))

        # Section-based chunking
        if parsed.sections and len(parsed.sections) > 1:
            for sec_title, sec_text in parsed.sections:
                if len(sec_text.strip()) < 30:
                    continue
                sec_chunks = self._split_text(sec_text)
                for sc in sec_chunks:
                    chunks.append(Chunk(
                        text=sc,
                        metadata={
                            **base_meta,
                            "chunk_type": "section",
                            "section": sec_title,
                        },
                    ))
        else:
            # No sections detected — split whole text
            for tc in self._split_text(content_source):
                chunks.append(Chunk(
                    text=tc,
                    metadata={**base_meta, "chunk_type": "body"},
                ))

        # Add title chunk at the front for better retrieval
        title_text = meta.title or parsed.title
        if title_text:
            title_chunk_text = f"TITLE: {title_text}"
            if meta.authors:
                title_chunk_text += f"\nAUTHORS: {', '.join(meta.authors)}"
            if meta.year:
                title_chunk_text += f"\nYEAR: {meta.year}"
            chunks.insert(0, Chunk(
                text=title_chunk_text,
                metadata={**base_meta, "chunk_type": "title"},
            ))

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split long text into overlapping chunks by token estimation."""
        if not text.strip():
            return []

        # Approximate tokens: 1 token ~= 4 characters for English, ~2 for CJK
        chars_per_token = 3
        chunk_chars = self.chunk_size * chars_per_token
        overlap_chars = self.chunk_overlap * chars_per_token

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_chars, text_len)
            if end <= start:
                break

            # Try to break at a paragraph or sentence boundary
            if end < text_len:
                # Look for paragraph break first
                for break_char in ["\n\n", "\n", ". ", "。", " "]:
                    pos = text.rfind(break_char, start, end)
                    if pos > start + chunk_chars // 2:
                        end = pos + len(break_char)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            next_start = end - overlap_chars
            if next_start <= start:
                next_start = end
            start = next_start
            if start >= text_len:
                break

        return chunks
