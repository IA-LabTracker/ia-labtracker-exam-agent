from typing import List, Optional
import os
from src.config import settings

_embed_model = None


def _get_local_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


class Embedder:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.EMBEDDINGS_PROVIDER
        if self.provider == "openai":
            import openai

            openai.api_key = settings.OPENAI_API_KEY

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "local":
            model = _get_local_model()
            return model.encode(texts, show_progress_bar=False).tolist()
        else:
            import openai

            resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
            return [e["embedding"] for e in resp.data]

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]
