"""FAISS-based retriever for user history items."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class FAISSRetriever:
    """
    FAISS-based retriever for user history items.

    Builds and queries a vector index for efficient top-k retrieval.
    Supports flat index (exact search) and IVF index (approximate search).
    """

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "flat",
        device: Optional[str] = None,
    ):
        """
        Args:
            encoder_name: Sentence transformer model for encoding items
            index_type: FAISS index type ("flat" or "ivf")
            device: Compute device for encoder
        """
        self.encoder_name = encoder_name
        self.index_type = index_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy load
        self._encoder = None
        self._tokenizer = None
        self._index = None
        self._items: list[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def _load_encoder(self) -> None:
        """Load encoder model."""
        if self._encoder is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self._encoder = AutoModel.from_pretrained(self.encoder_name).to(self.device)
        self._encoder.eval()

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        self._load_encoder()
        return self._encoder.config.hidden_size

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_encoder()

        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Batch encode
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        outputs = self._encoder(**encoded)

        # Mean pooling
        attention_mask = encoded["attention_mask"]
        hidden_states = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        # Normalize for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def _create_index(self, embeddings: np.ndarray):
        """Create FAISS index from embeddings."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        dim = embeddings.shape[1]
        n_items = embeddings.shape[0]

        if self.index_type == "flat":
            # Exact search using inner product (cosine with normalized vectors)
            self._index = faiss.IndexFlatIP(dim)
        elif self.index_type == "ivf":
            # Approximate search for larger datasets
            nlist = min(max(int(np.sqrt(n_items)), 4), 100)
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            # Train IVF index
            if n_items >= nlist:
                self._index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Add embeddings to index
        self._index.add(embeddings)

    def build_index(self, items: list[str]) -> None:
        """
        Build FAISS index from items.

        Args:
            items: List of text items to index
        """
        if not items:
            self._items = []
            self._embeddings = None
            self._index = None
            return

        self._items = list(items)
        self._embeddings = self._encode(self._items)
        self._create_index(self._embeddings)

    def add_items(self, items: list[str]) -> None:
        """Add items to existing index."""
        if not items:
            return

        new_embeddings = self._encode(items)

        if self._index is None:
            self._items = list(items)
            self._embeddings = new_embeddings
            self._create_index(new_embeddings)
        else:
            self._items.extend(items)
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
            self._index.add(new_embeddings)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[str] | tuple[list[str], list[float]]:
        """
        Retrieve top-k relevant items for query.

        Args:
            query: Query text
            top_k: Number of items to retrieve
            return_scores: Whether to return similarity scores

        Returns:
            List of retrieved items (and optionally scores)
        """
        if self._index is None or len(self._items) == 0:
            return ([], []) if return_scores else []

        # Encode query
        query_embedding = self._encode([query])

        # Search
        k = min(top_k, len(self._items))
        scores, indices = self._index.search(query_embedding, k)

        # Get items
        retrieved = [self._items[i] for i in indices[0] if i >= 0]

        if return_scores:
            score_list = [float(s) for s, i in zip(scores[0], indices[0]) if i >= 0]
            return retrieved, score_list

        return retrieved

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = 10,
    ) -> list[list[str]]:
        """
        Retrieve top-k items for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of items to retrieve per query

        Returns:
            List of retrieved item lists
        """
        if self._index is None or len(self._items) == 0:
            return [[] for _ in queries]

        # Encode queries
        query_embeddings = self._encode(queries)

        # Search
        k = min(top_k, len(self._items))
        _, indices = self._index.search(query_embeddings, k)

        # Get items for each query
        results = []
        for idx_row in indices:
            retrieved = [self._items[i] for i in idx_row if i >= 0]
            results.append(retrieved)

        return results

    def save(self, path: str) -> None:
        """Save retriever state to disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save index
        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))

        # Save items and metadata
        state = {
            "items": self._items,
            "encoder_name": self.encoder_name,
            "index_type": self.index_type,
            "embeddings": self._embeddings,
        }
        with open(path / "state.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load retriever state from disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        path = Path(path)

        # Load state
        with open(path / "state.pkl", "rb") as f:
            state = pickle.load(f)

        self._items = state["items"]
        self.encoder_name = state["encoder_name"]
        self.index_type = state["index_type"]
        self._embeddings = state["embeddings"]

        # Load index
        index_path = path / "index.faiss"
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))

    def __len__(self) -> int:
        """Return number of indexed items."""
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"FAISSRetriever(encoder={self.encoder_name}, "
            f"index_type={self.index_type}, items={len(self._items)})"
        )
