"""
FAISS-based similarity search engine for K-line patterns.

Handles vector storage, indexing, and efficient nearest neighbor search
for millions of K-line pattern embeddings.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pickle


class SimilaritySearchEngine:
    """
    FAISS-powered similarity search for K-line embeddings.

    Supports:
    - Adding vectors with metadata
    - Efficient nearest neighbor search
    - Index persistence (save/load)
    - Market-specific filtering
    """

    def __init__(
        self,
        dimension: int = 256,
        index_type: str = "flat",
        nlist: int = 4096,
    ):
        """
        Initialize search engine.

        Args:
            dimension: Embedding vector dimension
            index_type: 'flat' (exact) or 'ivf' (approximate, faster)
            nlist: Number of clusters for IVF index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist

        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._index_loaded = False
        self._total_vectors = 0

        # Try to import FAISS
        try:
            import faiss
            self.faiss = faiss
            self._faiss_available = True
        except ImportError:
            self._faiss_available = False
            print("Warning: FAISS not available. Using numpy fallback.")

    def _create_index(self):
        """Create a new FAISS index."""
        if not self._faiss_available:
            return

        if self.index_type == "flat":
            # Exact search using inner product (for normalized vectors = cosine similarity)
            self.index = self.faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            # Approximate search with IVF
            quantizer = self.faiss.IndexFlatIP(self.dimension)
            self.index = self.faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, self.faiss.METRIC_INNER_PRODUCT
            )
        elif self.index_type == "ivfpq":
            # Compressed index for large-scale (memory efficient)
            quantizer = self.faiss.IndexFlatL2(self.dimension)
            self.index = self.faiss.IndexIVFPQ(
                quantizer, self.dimension, self.nlist, 32, 8
            )

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ):
        """
        Add vectors with metadata to the index.

        Args:
            vectors: Numpy array of shape (n, dimension)
            metadata_list: List of metadata dicts for each vector
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors must match number of metadata entries")

        vectors = np.asarray(vectors, dtype=np.float32)

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        if self._faiss_available:
            if self.index is None:
                self._create_index()

            # Train index if needed (for IVF)
            if self.index_type in ("ivf", "ivfpq") and not self.index.is_trained:
                self.index.train(vectors)

            self.index.add(vectors)
        else:
            # Numpy fallback
            if not hasattr(self, "_vectors"):
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])

        self.metadata.extend(metadata_list)
        self._total_vectors += len(vectors)
        self._index_loaded = True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        markets: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            markets: Optional list of markets to filter results

        Returns:
            List of dicts with similarity_score and metadata
        """
        if not self._index_loaded:
            return []

        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # Search for more candidates if filtering by market
        search_k = top_k * 5 if markets else top_k

        if self._faiss_available:
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = min(128, self.nlist)

            distances, indices = self.index.search(query_vector, search_k)
            distances = distances[0]
            indices = indices[0]
        else:
            # Numpy fallback
            similarities = np.dot(self._vectors, query_vector.T).flatten()
            indices = np.argsort(similarities)[::-1][:search_k]
            distances = similarities[indices]

        # Build results with metadata
        results = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx].copy()

            # Filter by market if specified
            if markets and meta.get("market") not in markets:
                continue

            # Convert distance to similarity score
            # For inner product on normalized vectors, distance is already cosine similarity
            similarity_score = float(dist) if dist <= 1.0 else 1.0 / (1.0 + dist)

            results.append({
                "similarity_score": similarity_score,
                **meta,
            })

            if len(results) >= top_k:
                break

        return results

    def save_index(self, directory: str):
        """
        Save index and metadata to disk.

        Args:
            directory: Directory path to save files
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self._faiss_available and self.index is not None:
            self.faiss.write_index(self.index, str(dir_path / "index.faiss"))
        elif hasattr(self, "_vectors"):
            np.save(dir_path / "vectors.npy", self._vectors)

        # Save metadata
        with open(dir_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        # Save config
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "total_vectors": self._total_vectors,
        }
        with open(dir_path / "config.json", "w") as f:
            json.dump(config, f)

    def load_index(self, directory: str) -> bool:
        """
        Load index and metadata from disk.

        Args:
            directory: Directory path to load from

        Returns:
            True if loaded successfully
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            return False

        try:
            # Load config
            with open(dir_path / "config.json", "r") as f:
                config = json.load(f)

            self.dimension = config["dimension"]
            self.index_type = config["index_type"]
            self.nlist = config["nlist"]
            self._total_vectors = config["total_vectors"]

            # Load index
            if self._faiss_available and (dir_path / "index.faiss").exists():
                self.index = self.faiss.read_index(str(dir_path / "index.faiss"))
            elif (dir_path / "vectors.npy").exists():
                self._vectors = np.load(dir_path / "vectors.npy")

            # Load metadata
            with open(dir_path / "metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)

            self._index_loaded = True
            return True

        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def is_index_loaded(self) -> bool:
        """Check if index is loaded and ready."""
        return self._index_loaded

    def get_total_vectors(self) -> int:
        """Get total number of indexed vectors."""
        return self._total_vectors

    def get_indexed_markets(self) -> List[str]:
        """Get list of markets in the index."""
        markets = set()
        for meta in self.metadata:
            if "market" in meta:
                markets.add(meta["market"])
        return list(markets)

    def clear(self):
        """Clear the index."""
        self.index = None
        self.metadata = []
        self._index_loaded = False
        self._total_vectors = 0
        if hasattr(self, "_vectors"):
            del self._vectors


class HybridSearchEngine:
    """
    Hybrid search combining FAISS and DTW.

    First uses FAISS for fast approximate search, then refines
    top candidates using DTW for more accurate matching.
    """

    def __init__(self, faiss_engine: SimilaritySearchEngine):
        """
        Initialize hybrid search.

        Args:
            faiss_engine: FAISS search engine instance
        """
        self.faiss_engine = faiss_engine

    def search(
        self,
        query_vector: np.ndarray,
        query_series: np.ndarray,
        top_k: int = 10,
        faiss_candidates: int = 50,
        markets: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with FAISS + DTW refinement.

        Args:
            query_vector: Query embedding vector
            query_series: Raw time series for DTW
            top_k: Final number of results
            faiss_candidates: Number of FAISS candidates for DTW refinement
            markets: Optional market filter

        Returns:
            Refined search results
        """
        from app.services.dtw_matcher import DTWMatcher

        # Step 1: FAISS search for candidates
        candidates = self.faiss_engine.search(
            query_vector, top_k=faiss_candidates, markets=markets
        )

        if not candidates:
            return []

        # Step 2: DTW refinement
        dtw_matcher = DTWMatcher()
        refined = dtw_matcher.refine_matches(
            query_series=query_series,
            candidates=candidates,
            top_k=top_k,
        )

        return refined
