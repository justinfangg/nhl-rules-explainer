"""Local sentence-transformers embeddings."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from nhl_rules.config import get_settings

# Some retrieval models expect queries (not passages) to carry an instruction prefix.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


@lru_cache
def load_model(name: str | None = None) -> SentenceTransformer:
    return SentenceTransformer(name or get_settings().embedding_model)


def embed_passages(texts: list[str], model_name: str | None = None):
    """Embed documents (rule text). Returns a 2-D numpy array of unit vectors."""
    model = load_model(model_name)
    return model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def embed_query(text: str, model_name: str | None = None):
    """Embed a search query. Returns a 1-D numpy unit vector."""
    name = model_name or get_settings().embedding_model
    return load_model(name).encode(QUERY_PREFIXES.get(name, "") + text, normalize_embeddings=True)
