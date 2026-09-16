"""Ollama embedding access with a process-wide (model, text) -> vector cache.

Design note (brain.md D3/D8): facts are embedded *lazily* on first retrieval, and
the query string is embedded through the same cache so repeated routine queries
cost nothing after the first embedding. All vectors are cached in memory only;
there is no persistence of embeddings across restarts.
"""

import threading
from typing import List, Optional, Union

import requests

_cache = {}
_lock = threading.Lock()


def embed_ollama(
    texts: Union[str, List[str]],
    model: str = "nomic-embed-text",
    endpoint: str = "http://localhost:11434",
) -> Optional[Union[list, List[list]]]:
    """Embed one string or a list of strings.

    Returns a single vector for a str input, a list of vectors for a list input,
    or None entries for any text that failed to embed. Misses are batch-fetched
    in a single /api/embed call; hits are served from the shared cache.
    """
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
    out: List[Optional[list]] = [None] * len(items)

    missing_idx: List[int] = []
    missing_texts: List[str] = []
    with _lock:
        for i, t in enumerate(items):
            key = (model, t)
            if key in _cache:
                out[i] = _cache[key]
            else:
                missing_idx.append(i)
                missing_texts.append(t)

    if missing_texts:
        try:
            resp = requests.post(
                f"{endpoint}/api/embed",
                json={"model": model, "input": missing_texts},
                timeout=60,
            )
            resp.raise_for_status()
            vectors = resp.json().get("embeddings", [])
        except Exception:
            vectors = []
        with _lock:
            for k, i in enumerate(missing_idx):
                v = vectors[k] if k < len(vectors) else None
                _cache[(model, missing_texts[k])] = v
                out[i] = v

    return out[0] if single else out


def cached_vector(text: str, model: str = "nomic-embed-text"):
    with _lock:
        return _cache.get((model, text))