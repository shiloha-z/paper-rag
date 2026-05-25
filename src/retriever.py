from .config import RetrievalConfig
from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    """Search the paper knowledge base with embedding-based retrieval."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, config: RetrievalConfig):
        self.embedder = embedder
        self.store = vector_store
        self.top_k = config.top_k
        self.min_score = config.min_score

    def search(self, query: str, top_k: int | None = None, filter_collection: str | None = None) -> list[dict]:
        """Search for paper chunks relevant to the query.

        Args:
            query: Natural language query
            top_k: Override default top_k
            filter_collection: Optional Zotero collection name filter
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.embed_one(query)

        hits = self.store.search(query_embedding, top_k=k, min_score=self.min_score)

        # Optional collection filter (post-filter on metadata)
        if filter_collection:
            hits = [
                h for h in hits
                if filter_collection.lower() in h.get("metadata", {}).get("collections", "").lower()
            ]

        return hits

    def search_by_paper(self, query: str, top_papers: int = 5) -> dict[str, list[dict]]:
        """Search and group results by paper title."""
        hits = self.search(query, top_k=self.top_k * 2)

        grouped: dict[str, list[dict]] = {}
        for h in hits:
            title = h.get("metadata", {}).get("title", "Unknown")
            if title not in grouped:
                grouped[title] = []
            if len(grouped[title]) < 3:  # max 3 chunks per paper
                grouped[title].append(h)

        # Sort groups by best chunk score
        sorted_groups = sorted(
            grouped.items(),
            key=lambda kv: max(ch["score"] for ch in kv[1]),
            reverse=True,
        )
        return dict(sorted_groups[:top_papers])

    def format_hits(self, hits: list[dict]) -> str:
        """Format search results for display or LLM context."""
        if not hits:
            return "No matching papers found."

        parts = []
        for i, h in enumerate(hits):
            m = h.get("metadata", {})
            header = f"[{i+1}] {m.get('title', 'Unknown')}"
            if m.get("authors"):
                header += f" — {m['authors']}"
            if m.get("year"):
                header += f" ({m['year']})"
            if m.get("chunk_type") == "abstract":
                header += " [ABSTRACT]"

            parts.append(f"{header}\nScore: {h['score']}\n{h['text'][:600]}")

        return "\n\n---\n\n".join(parts)
