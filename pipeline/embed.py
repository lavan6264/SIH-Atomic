"""sentence-transformers embedding logic + Chroma vector store.

Embeddings are computed on normalized_description text (see
pipeline/normalize.py) so that fragmented raw ERP strings from different
CPSEs land close together in vector space once cleaned to a shared
vocabulary. find_similar() normalizes its query the same way, so a raw
(un-normalized) query behaves the way a new incoming ERP line item would.
"""

from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from pipeline.normalize import normalize_description

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "materials"
DB_DIR = Path(__file__).resolve().parent.parent / "db" / "chroma"

_model = None
_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    """Load (once) and return the shared sentence-transformers model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list) -> np.ndarray:
    """Embed a list of strings into L2-normalized vectors."""
    model = get_embedder()
    embeddings = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings)


def _get_client():
    global _client
    if _client is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(DB_DIR))
    return _client


def get_collection():
    """Open (or create) the persisted 'materials' collection, cached."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def build_vector_store(df: pd.DataFrame):
    """Embed every normalized_description in df and upsert into Chroma.

    Expects df to have cpse, material_id, raw_description, and
    normalized_description columns. material_id must be globally unique
    across all 4 CPSEs (it is, by construction: "<CPSE>-00001" etc.) since
    it is used as the Chroma document id. Uses upsert so re-running this
    on the same data is idempotent. Returns the collection.
    """
    collection = get_collection()

    texts = df["normalized_description"].fillna("").tolist()
    embeddings = embed_texts(texts)

    ids = df["material_id"].astype(str).tolist()
    metadatas = df[["cpse", "material_id", "raw_description", "normalized_description"]].astype(str).to_dict("records")

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=texts,
    )
    return collection


def find_similar(query_text: str, k: int = 5) -> list:
    """Return the top-k most similar stored items to query_text.

    query_text is normalized the same way stored items were before
    embedding, so a raw ERP-style string can be passed directly. Each
    result dict has: material_id, cpse, raw_description,
    normalized_description, similarity (cosine similarity, higher = closer,
    computed as 1 - cosine distance since the collection is configured
    for hnsw:space="cosine").
    """
    collection = get_collection()
    normalized_query = normalize_description(query_text)
    query_embedding = embed_texts([normalized_query])[0]

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
    )

    matches = []
    for id_, distance, metadata in zip(result["ids"][0], result["distances"][0], result["metadatas"][0]):
        matches.append({
            "material_id": id_,
            "cpse": metadata.get("cpse"),
            "raw_description": metadata.get("raw_description"),
            "normalized_description": metadata.get("normalized_description"),
            "similarity": 1.0 - distance,
        })
    return matches
