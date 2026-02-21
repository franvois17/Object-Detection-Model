"""FAISS-based embedding index for cosine-similarity search."""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


class EmbeddingIndex:
    """Flat inner-product index over L2-normalised vectors (= cosine similarity).

    Each vector is associated with a ``product_id`` label that is stored
    in a parallel list so that search results can be mapped back to
    products.
    """

    def __init__(self, dim: int = 1280) -> None:
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._labels: list[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise each row so inner product equals cosine similarity."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # avoid division by zero
        return vectors / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, labels: list[int]) -> None:
        """Add *embeddings* to the index with corresponding *labels*.

        Parameters
        ----------
        embeddings : np.ndarray
            Array of shape ``(N, dim)`` or ``(dim,)`` for a single vector.
        labels : list[int]
            One ``product_id`` per embedding vector.
        """
        embeddings = self._normalize(embeddings)
        if embeddings.shape[0] != len(labels):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) does not "
                f"match number of labels ({len(labels)})"
            )
        self._index.add(embeddings)
        self._labels.extend(labels)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Search for the *k* nearest neighbours of *query*.

        Parameters
        ----------
        query : np.ndarray
            1-D array of shape ``(dim,)`` or 2-D ``(1, dim)``.
        k : int
            Number of neighbours to return.

        Returns
        -------
        list[tuple[int, float]]
            List of ``(product_id, similarity_score)`` sorted by
            descending similarity.
        """
        if len(self) == 0:
            return []
        query = self._normalize(query)
        k = min(k, len(self))
        distances, indices = self._index.search(query, k)
        results: list[tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((self._labels[idx], float(dist)))
        return results

    def clear(self) -> None:
        """Remove all vectors and labels from the index."""
        self._index.reset()
        self._labels.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the FAISS index and label map to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dim": self.dim,
            "index_bytes": faiss.serialize_index(self._index),
            "labels": self._labels,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        """Deserialise a previously saved index from *path*."""
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.dim = data["dim"]
        self._index = faiss.deserialize_index(data["index_bytes"])
        self._labels = data["labels"]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._index.ntotal
