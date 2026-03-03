from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import get_settings
from src.utils.logging import logger


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        logger.info("Loaded local model: %s", model_name)

    def embed(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info("Using OpenAI embedder: %s", model)

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
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
