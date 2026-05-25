import time
import os
from openai import OpenAI

from .config import EmbeddingConfig


class APIEmbedder:
    """Generate embeddings via OpenAI-compatible API with batching and retry."""

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "text-embedding-3-small",
        batch_size: int = 20,
        dimensions: int | None = None,
    ):
        if not api_key:
            raise ValueError("Embedding API key is required. Set in config.yaml or PAPERRAG_EMBED_API_KEY env var.")

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_embeddings.extend(self._embed_batch(batch))

        return all_embeddings

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        kwargs = {"model": self.model, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(**kwargs)
                return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Embedding failed after {max_retries} retries: {e}")
        return []


class LocalEmbedder:
    """Generate embeddings with a local SentenceTransformer model (e.g. BGE-M3)."""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 8,
        device: str = "",
    ):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Local embedding model not found: {model_path}")

        from sentence_transformers import SentenceTransformer

        if not device:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_path, device=device)
        self.batch_size = batch_size
        self._device = device

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # normalize_embeddings=True gives unit vectors for cosine similarity
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def create_embedder(config: EmbeddingConfig) -> APIEmbedder | LocalEmbedder:
    """Factory: return the right embedder based on config.provider."""
    provider = config.provider.lower()

    if provider == "local":
        return LocalEmbedder(
            model_path=config.model_path,
            batch_size=config.batch_size,
            device=config.device,
        )

    if provider == "api":
        return APIEmbedder(
            api_base=config.api_base,
            api_key=config.api_key,
            model=config.model,
            batch_size=config.batch_size,
            dimensions=config.dimensions,
        )

    raise ValueError(
        f"Unknown embedding provider: {provider}. Use 'local' or 'api'."
    )


# Keep backward-compatible alias
Embedder = APIEmbedder
