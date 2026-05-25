from pathlib import Path
import hashlib

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunker import Chunk


class VectorStore:
    """ChromaDB-backed vector store for paper chunks."""

    def __init__(self, persist_path: str = "./data/chroma", collection_name: str = "papers"):
        self.persist_path = str(Path(persist_path).resolve())
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Add chunk-embedding pairs to the store. Returns count added."""
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeds = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            source = (
                chunk.metadata.get("item_key")
                or chunk.metadata.get("doi")
                or chunk.metadata.get("title")
                or "unknown"
            )
            identity = "|".join([
                str(source),
                str(chunk.metadata.get("chunk_type", "body")),
                str(chunk.metadata.get("section", "")),
                chunk.text,
            ])
            digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
            prefix = f"{source}_{chunk.metadata.get('chunk_type', 'body')}"
            prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix)[:180]
            chunk_id = f"{prefix}_{digest}_{i}"

            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append(_sanitize_metadata(chunk.metadata))
            embeds.append(emb)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeds,
        )
        return len(ids)

    def search(self, query_embedding: list[float], top_k: int = 8, min_score: float = 0.0) -> list[dict]:
        """Search by embedding vector, return top-k results with metadata."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            # Cosine distance → similarity
            similarity = 1.0 - distance

            if similarity < min_score:
                continue

            hits.append({
                "id": doc_id,
                "text": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "score": round(similarity, 4),
            })

        return hits

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        """Delete all entries from the collection."""
        self.client.delete_collection(self._collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def stats(self) -> dict:
        """Return collection statistics."""
        return {
            "name": self._collection_name,
            "count": self.collection.count(),
            "path": self.persist_path,
        }


def _sanitize_metadata(meta: dict) -> dict:
    """ChromaDB only allows str, int, float, bool metadata values."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean
