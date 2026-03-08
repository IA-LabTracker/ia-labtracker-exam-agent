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
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        super().__init__()
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info("Using OpenAI embedder: %s", model)

    def _embed_uncached(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return resp.data[0].embedding

    def _embed_batch_uncached(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


class Embedder:
    @staticmethod
    def create(settings=None) -> BaseEmbedder:
        settings = settings or get_settings()
        if settings.embeddings_provider == "openai":
            return OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
            )
        return LocalEmbedder(model_name=settings.embedding_model)
