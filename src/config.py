import os
import re
from pathlib import Path
from dataclasses import dataclass, field

import yaml


@dataclass
class EmbeddingConfig:
    provider: str = "local"      # "local" or "api"
    model: str = "text-embedding-3-small"
    model_path: str = ""         # path for local models (e.g. BGE-M3)
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    batch_size: int = 8
    dimensions: int | None = None
    device: str = ""             # "cpu", "cuda", or "" for auto-detect


@dataclass
class LLMConfig:
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2048


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150
    keep_abstract_separate: bool = True
    keep_metadata: bool = True


@dataclass
class VectorStoreConfig:
    path: str = "./data/chroma"
    collection: str = "papers"


@dataclass
class RetrievalConfig:
    top_k: int = 8
    rerank: bool = False
    min_score: float = 0.0


@dataclass
class PDFConfig:
    max_pages_per_paper: int = 100


@dataclass
class ZoteroConfig:
    data_dir: str = ""
    profile: str = ""


@dataclass
class PDFDirConfig:
    path: str = ""
    recursive: bool = True


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    zotero: ZoteroConfig = field(default_factory=ZoteroConfig)
    pdf_dir: PDFDirConfig = field(default_factory=PDFDirConfig)


def _resolve_env(value: str) -> str:
    """Resolve ${VAR} and $VAR patterns in a string."""
    if not isinstance(value, str):
        return value

    def _replace(match):
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, "")

    return re.sub(r"\$\{(\w+)\}|\$(\w+)", _replace, value)


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert dict to dataclass, resolving env vars."""
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, val in data.items():
        if key in field_types:
            if isinstance(val, str):
                val = _resolve_env(val)
            if isinstance(val, dict) and hasattr(field_types[key], "__dataclass_fields__"):
                kwargs[key] = _dict_to_dataclass(field_types[key], val)
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str = "config.yaml") -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return _dict_to_dataclass(Config, data)
