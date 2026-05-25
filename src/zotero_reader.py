import os
import re
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PaperAttachment:
    """A PDF attachment of a Zotero item."""
    item_key: str
    filename: str
    storage_path: str   # absolute path to the PDF file


@dataclass
class PaperMeta:
    """Metadata for a single paper from Zotero."""
    item_key: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""
    url: str = ""
    tags: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    date_added: str = ""
    attachments: list[PaperAttachment] = field(default_factory=list)


def _find_zotero_data_dir() -> Path | None:
    """Detect Zotero data directory on Windows/Mac/Linux."""
    home = Path.home()

    candidates = []

    if os.name == "nt":
        candidates = [
            home / "Zotero",                                        # Zotero 7
            Path(os.environ.get("APPDATA", "")) / "Zotero" / "Zotero",  # Zotero 6
            Path(os.environ.get("LOCALAPPDATA", "")) / "Zotero" / "Zotero",
        ]
    elif os.uname().sysname == "Darwin":
        candidates = [
            home / "Library" / "Application Support" / "Zotero",
        ]
    else:
        candidates = [
            home / ".zotero" / "zotero",
            home / "Zotero",
        ]

    for p in candidates:
        if p.exists():
            return p
    return None


def _find_profile_dir(zotero_data: Path, preferred: str = "") -> Path | None:
    """Find the Zotero profile directory (contains zotero.sqlite)."""
    profiles_dir = zotero_data / "Profiles"
    if not profiles_dir.exists():
        return None

    if preferred:
        preferred_path = profiles_dir / preferred
        if preferred_path.exists():
            return preferred_path

    for entry in sorted(profiles_dir.iterdir()):
        db = entry / "zotero.sqlite"
        if db.exists():
            return entry
    return None


def _sanitize_key(key: str) -> str:
    """Clean Zotero item key — keep only alphanumeric chars."""
    return re.sub(r"[^a-zA-Z0-9]", "", key)


def _get_storage_path(zotero_data: Path, profile_dir: Path, item_key: str) -> str:
    """Resolve the storage directory for a given item key."""
    storage = zotero_data / "storage" / item_key
    if storage.exists():
        return str(storage)
    return ""


class ZoteroReader:
    """Read papers from a local Zotero SQLite database."""

    def __init__(self, data_dir: str = "", profile: str = ""):
        self.data_dir = Path(data_dir) if data_dir else None
        self.profile_name = profile

        if self.data_dir is None:
            self.data_dir = _find_zotero_data_dir()

        if self.data_dir is None:
            raise FileNotFoundError(
                "Cannot find Zotero data directory. "
                "Set zotero.data_dir in config.yaml to the Zotero folder."
            )

        # Zotero 7: zotero.sqlite may be directly in data_dir
        direct_db = self.data_dir / "zotero.sqlite"
        if direct_db.exists():
            self.db_path = direct_db
            self.profile_dir = self.data_dir
        else:
            # Zotero 6: zotero.sqlite is in Profiles/<profile>/
            self.profile_dir = _find_profile_dir(self.data_dir, profile)
            if self.profile_dir is None:
                raise FileNotFoundError(
                    f"No Zotero profile with zotero.sqlite found in {self.data_dir}"
                )
            self.db_path = self.profile_dir / "zotero.sqlite"
            if not self.db_path.exists():
                raise FileNotFoundError(f"zotero.sqlite not found at {self.db_path}")

        # Copy DB to temp to avoid lock when Zotero is running
        import tempfile
        import shutil
        self._temp_dir = tempfile.mkdtemp(prefix="paperrag_")
        self._temp_db = Path(self._temp_dir) / "zotero.sqlite"
        shutil.copy2(str(self.db_path), str(self._temp_db))
        self.db_path = self._temp_db

    def __del__(self):
        """Clean up temp copy of database."""
        import shutil
        if hasattr(self, "_temp_dir") and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def get_all_papers(self) -> list[PaperMeta]:
        """Extract all journal articles and conference papers with metadata."""
        papers: list[PaperMeta] = []

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()

            # Get all top-level items — item type is on items.itemTypeID, not in itemData
            cursor.execute("""
                SELECT i.itemID, i.key as item_key, it.typeName as item_type
                FROM items i
                JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                WHERE it.typeName IN (
                    'journalArticle', 'conferencePaper', 'preprint',
                    'thesis', 'bookSection', 'book'
                )
            """)

            item_rows = cursor.fetchall()

            for row in item_rows:
                item_id = row["itemID"]
                item_key = row["item_key"]

                meta = self._read_item_meta(conn, item_id, item_key)
                meta.attachments = self._read_attachments(conn, item_id, item_key)
                if meta.title:
                    papers.append(meta)

        finally:
            conn.close()

        return papers

    def _read_item_meta(self, conn: sqlite3.Connection, item_id: int, item_key: str) -> PaperMeta:
        cursor = conn.cursor()

        # Map Zotero field names -> PaperMeta attributes via itemData
        cursor.execute("""
            SELECT f.fieldName, idv.value
            FROM itemData id
            JOIN fields f ON id.fieldID = f.fieldID
            JOIN itemDataValues idv ON id.valueID = idv.valueID
            WHERE id.itemID = ?
        """, (item_id,))

        raw = {row["fieldName"]: row["value"] for row in cursor.fetchall()}

        authors = []
        if raw.get("firstCreator"):
            authors.append(raw["firstCreator"])
        # Additional creators come through creators table

        # Read all creators
        cursor.execute("""
            SELECT c.firstName, c.lastName, c.fieldMode, ct.creatorType
            FROM creators c
            JOIN itemCreators ic ON c.creatorID = ic.creatorID
            JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
            WHERE ic.itemID = ?
            ORDER BY ic.orderIndex
        """, (item_id,))

        creator_rows = cursor.fetchall()
        parsed_authors = []
        for cr in creator_rows:
            if cr["creatorType"] == "author":
                name = f"{cr['lastName']}, {cr['firstName']}".strip(", ")
                parsed_authors.append(name)
        if not parsed_authors and raw.get("firstCreator"):
            parsed_authors.append(raw["firstCreator"])

        # Read tags
        cursor.execute("""
            SELECT t.name FROM tags t
            JOIN itemTags it ON t.tagID = it.tagID
            WHERE it.itemID = ?
        """, (item_id,))
        tags = [r["name"] for r in cursor.fetchall()]

        # Read collections
        cursor.execute("""
            SELECT c.collectionName FROM collections c
            JOIN collectionItems ci ON c.collectionID = ci.collectionID
            WHERE ci.itemID = ?
        """, (item_id,))
        collections = [r["collectionName"] for r in cursor.fetchall()]

        year = raw.get("date", "")[:4] if raw.get("date") else ""

        return PaperMeta(
            item_key=item_key,
            title=raw.get("title", ""),
            authors=parsed_authors,
            abstract=raw.get("abstractNote", ""),
            year=year,
            journal=(raw.get("publicationTitle") or raw.get("journalAbbreviation") or ""),
            doi=raw.get("DOI", ""),
            url=raw.get("url", ""),
            tags=tags,
            collections=collections,
            date_added=raw.get("dateAdded", ""),
        )

    def _read_attachments(self, conn: sqlite3.Connection, item_id: int, item_key: str) -> list[PaperAttachment]:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ia.itemID as attachment_item_id, i.key as attachment_key,
                   ia.path as rel_path
            FROM itemAttachments ia
            JOIN items i ON ia.itemID = i.itemID
            WHERE ia.parentItemID = ?
              AND ia.contentType = 'application/pdf'
        """, (item_id,))

        attachments = []
        for att_row in cursor.fetchall():
            att_key = att_row["attachment_key"]
            rel_path = att_row["rel_path"] or ""

            # Resolve PDF path: try storage/<key>/path first, then storage/<key>/
            storage_dir = self.data_dir / "storage" / att_key
            pdf_path = ""
            if storage_dir.exists():
                if rel_path:
                    candidate = storage_dir / rel_path
                    if candidate.exists():
                        pdf_path = str(candidate)
                if not pdf_path:
                    # Find any PDF in the storage dir
                    for f in storage_dir.iterdir():
                        if f.suffix.lower() == ".pdf":
                            pdf_path = str(f)
                            break

            attachments.append(PaperAttachment(
                item_key=item_key,
                filename=rel_path or f"{att_key}.pdf",
                storage_path=pdf_path,
            ))

        return attachments


def scan_pdf_directory(directory: str, recursive: bool = True) -> list[PaperMeta]:
    """Scan a directory for PDF files and create stub PaperMeta entries."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(dir_path.glob(pattern))

    papers = []
    for pdf_path in pdf_files:
        name = pdf_path.stem
        papers.append(PaperMeta(
            item_key=pdf_path.as_posix(),  # use path as key
            title=name,
            attachments=[PaperAttachment(
                item_key=pdf_path.as_posix(),
                filename=pdf_path.name,
                storage_path=str(pdf_path),
            )],
        ))

    return papers
