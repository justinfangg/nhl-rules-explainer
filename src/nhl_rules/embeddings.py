"""Local sentence-transformers embeddings."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from nhl_rules.config import get_settings


@lru_cache
def load_model(name: str | None = None) -> SentenceTransformer:
    return SentenceTransformer(name or get_settings().embedding_model)


def embed_passages(texts: list[str], model_name: str | None = None):
    """Embed documents (rule text). Returns a 2-D numpy array of unit vectors."""
    model = load_model(model_name)
    return model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
