from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import get_settings
from src.utils.logging import logger


class BaseEmbedder(ABC):
    """Base class with built-in embedding cache.

    Caches embed() and embed_batch() results by text to avoid
    re-computing embeddings for the same query strings.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        key = text.strip().lower()
        if key in self._cache:
            return self._cache[key]
        result = self._embed_uncached(text)
        self._cache[key] = result
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, t in enumerate(texts):
            key = t.strip().lower()
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            new_embeddings = self._embed_batch_uncached(uncached_texts)
            for idx, t, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                key = t.strip().lower()
                self._cache[key] = emb
                results[idx] = emb

        return results  # type: ignore[return-value]

    @abstractmethod
    def _embed_uncached(self, text: str) -> list[float]: ...

    @abstractmethod
    def _embed_batch_uncached(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        logger.info("Loaded local model: %s", model_name)

    def _embed_uncached(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()

    def _embed_batch_uncached(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings with optional dimension truncation.

    Uses the `dimensions` parameter supported by text-embedding-3-* models
    to produce vectors that match the DB schema (e.g. 768-dim) without
    needing to alter the database.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int | None = None,
    ):
        super().__init__()
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        dim_str = f" (dimensions={dimensions})" if dimensions else ""
        logger.info("Using OpenAI embedder: %s%s", model, dim_str)

    def _embed_uncached(self, text: str) -> list[float]:
        kwargs: dict = {"model": self._model, "input": [text]}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        resp = self._client.embeddings.create(**kwargs)
        return resp.data[0].embedding

    def _embed_batch_uncached(self, texts: list[str]) -> list[list[float]]:
        # OpenAI API supports up to 2048 inputs per call; batch in chunks of 100
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), 100):
            chunk = texts[i : i + 100]
            kwargs: dict = {"model": self._model, "input": chunk}
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            resp = self._client.embeddings.create(**kwargs)
            # OpenAI may return results out of order — sort by index
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            all_embeddings.extend(d.embedding for d in sorted_data)
        return all_embeddings


class Embedder:
    @staticmethod
    def create(settings=None) -> BaseEmbedder:
        settings = settings or get_settings()
        if settings.embeddings_provider == "openai":
            return OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
                dimensions=settings.embedding_dim if settings.embedding_dim else None,
            )
        return LocalEmbedder(model_name=settings.embedding_model)
