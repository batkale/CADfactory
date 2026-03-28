"""
RAG (Retrieval-Augmented Generation) store for CadQuery examples.

Stores user-approved (thumbs-up) generation results as vector embeddings
so that future generations can retrieve the most similar past successes
and inject them as few-shot examples into the system prompt.

Uses:
  - chromadb  : local persistent vector store (no server required)
  - Gemini text-embedding-004 : for embedding descriptions

Storage location: ./rag_store/  (configurable via RAG_STORE_DIR env var)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_RAG_DIR = os.environ.get("RAG_STORE_DIR", "./rag_store")
_EMBED_MODEL = "gemini-embedding-001"
_COLLECTION_NAME = "cad_examples"

# Lazy-initialised singletons
_client = None
_collection = None
_gemini_client = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        os.makedirs(_RAG_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=_RAG_DIR)
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"RAG store ready at {_RAG_DIR} ({_collection.count()} examples)")
    except ImportError:
        logger.warning("chromadb not installed — RAG store disabled. Run: pip install chromadb")
        _collection = None
    except Exception as e:
        logger.error(f"RAG store init failed: {e}")
        _collection = None
    return _collection


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    try:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        _gemini_client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Gemini client init failed: {e}")
    return _gemini_client


def _embed(text: str) -> list[float] | None:
    """Embed text using Gemini text-embedding-004. Returns None on failure."""
    client = _get_gemini_client()
    if client is None:
        return None
    try:
        result = client.models.embed_content(
            model=_EMBED_MODEL,
            contents=text,
        )
        return result.embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def add_example(part_id: int, description: str, script: str, manufacturing_method: str = "fdm") -> bool:
    """
    Add a user-approved (thumbs-up) example to the RAG store.
    Returns True on success.
    """
    collection = _get_collection()
    if collection is None:
        return False

    doc_id = f"part_{part_id}"

    # Check if already exists — update instead of duplicate
    existing = collection.get(ids=[doc_id])
    if existing["ids"]:
        collection.update(
            ids=[doc_id],
            documents=[description],
            metadatas=[{"script": script, "method": manufacturing_method, "part_id": part_id}],
        )
        logger.info(f"RAG store: updated example for part {part_id}")
        return True

    embedding = _embed(description)
    if embedding is None:
        # Fall back to storing without embedding (chromadb will use its default)
        collection.add(
            ids=[doc_id],
            documents=[description],
            metadatas=[{"script": script, "method": manufacturing_method, "part_id": part_id}],
        )
    else:
        collection.add(
            ids=[doc_id],
            documents=[description],
            embeddings=[embedding],
            metadatas=[{"script": script, "method": manufacturing_method, "part_id": part_id}],
        )

    logger.info(f"RAG store: added example for part {part_id} (total: {collection.count()})")
    return True


def remove_example(part_id: int) -> bool:
    """Remove a downvoted or deleted example from the RAG store."""
    collection = _get_collection()
    if collection is None:
        return False
    try:
        collection.delete(ids=[f"part_{part_id}"])
        logger.info(f"RAG store: removed example for part {part_id}")
        return True
    except Exception as e:
        logger.error(f"RAG remove failed: {e}")
        return False


def get_similar_examples(
    description: str,
    manufacturing_method: str = "fdm",
    k: int = 3,
) -> list[dict]:
    """
    Retrieve the top-k most similar approved examples for a given description.

    Returns list of {"description": str, "script": str, "method": str} dicts,
    best match first. Returns [] if RAG store is empty or unavailable.
    """
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return []

    embedding = _embed(description)
    try:
        if embedding is not None:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        else:
            # Text-based fallback if embedding fails
            results = collection.query(
                query_texts=[description],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return []

    examples = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # cosine distance: 0 = identical, 2 = opposite. Skip poor matches (>1.2).
        if dist > 1.2:
            continue
        examples.append({
            "description": doc,
            "script": meta.get("script", ""),
            "method": meta.get("method", "fdm"),
        })

    return examples


def store_count() -> int:
    """Return number of examples currently in the RAG store."""
    collection = _get_collection()
    if collection is None:
        return 0
    return collection.count()
