from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedPaper:
    """Result of parsing a paper PDF."""
    item_key: str
    title: str
    authors: list[str]
    abstract: str
    full_text: str
    pages: int
    sections: list[tuple[str, str]]   # [(section_title, section_text), ...]


class PDFParser:
    """Extract text from PDFs using PyMuPDF."""

    def __init__(self, max_pages: int = 100):
        self.max_pages = max_pages

    def parse(self, pdf_path: str) -> ParsedPaper:
        """Parse a single PDF and return structured text."""
        import fitz  # PyMuPDF — lazy import so the CLI can show nicer errors

        doc = fitz.open(pdf_path)
        pages = min(doc.page_count, self.max_pages)

        full_text_parts = []
        for i in range(pages):
            page = doc[i]
            text = page.get_text("text")
            full_text_parts.append(text)

        doc.close()
        full_text = "\n\n".join(full_text_parts)
        sections = self._split_sections(full_text)
        abstract = self._extract_abstract(full_text, sections)

        return ParsedPaper(
            item_key="",
            title="",
            authors=[],
            abstract=abstract,
            full_text=full_text,
            pages=pages,
            sections=sections,
        )

    def _extract_abstract(self, full_text: str, sections: list[tuple[str, str]]) -> str:
        """Try to locate the abstract section."""
        for sec_title, sec_text in sections:
            title_lower = sec_title.lower().strip()
            if title_lower in ("abstract", "abstract.", "summary"):
                return sec_text.strip()

        # Fallback: look for "Abstract" keyword in the first 3000 chars
        head = full_text[:3000]
        import re
        m = re.search(
            r"(?:^|\n)abstract[\.\-\s:]*(.+?)(?:\n\n|\n(?:introduction|1\.?\s))",
            head, re.IGNORECASE | re.DOTALL
        )
        if m:
            return m.group(1).strip()

        # Last resort: first paragraph before any section header
        first_sec = sections[0][1] if sections else full_text[:2000]
        return first_sec.strip()[:2000]

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Split paper into rough sections by common header patterns."""
        import re

        patterns = [
            # Numbered: "1. Introduction", "2. Related Work"
            r"(?=\n\d+\.?\s+[A-Z][A-Za-z\s\-]+(?:\n|$))",
            # Named: "Introduction\n", "METHODS\n"
            r"(?=\n(?:Abstract|Introduction|Related\s+Work|Background|Method|Approach|"
            r"Experiments?|Evaluation|Results?|Discussion|Conclusion|References|"
            r"Acknowledgments?|Appendix)\s*\n)",
        ]

        combined = "|".join(f"({p})" for p in patterns)
        parts = re.split(combined, text, flags=re.IGNORECASE)

        if len(parts) <= 1:
            return [("full", text)]

        sections = []
        current_title = "preamble"
        current_text = ""

        for part in parts:
            if not part:
                continue
            stripped = part.strip()
            # Check if this is a section header (short line, all caps or title case)
            lines = stripped.split("\n")
            first_line = lines[0].strip() if lines else ""

            is_header = (
                len(lines) <= 3
                and len(first_line) < 100
                and (
                    first_line.isupper()
                    or first_line.istitle()
                    or bool(re.match(r"^\d+\.?\s+[A-Z]", first_line))
                )
            )

            if is_header and len(stripped) < 120:
                if current_text.strip():
                    sections.append((current_title, current_text.strip()))
                current_title = first_line
                rest = "\n".join(lines[1:]) if len(lines) > 1 else ""
                current_text = rest
            else:
                current_text += "\n" + stripped

        if current_text.strip():
            sections.append((current_title, current_text.strip()))

        return sections if sections else [("full", text)]
