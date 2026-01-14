"""Retrieval-Augmented Personalized LLM."""

from __future__ import annotations

from typing import Optional

import torch

from personalize.models.base import GenerationOutput, PersonalizedLLM
from personalize.models.e2p import E2PLLM
from personalize.models.persoma_llm import PERSOMALLM
from personalize.retrieval import FAISSRetriever


class RetrievalAugmentedLLM(PersonalizedLLM):
    """
    Retrieval-Augmented Personalization.

    Uses FAISS to retrieve relevant history items, then passes
    them to an underlying encoder (E2P or PERSOMA).

    Benefits:
    - Scales to users with very long histories (1000+ items)
    - Retrieves only query-relevant context
    - Combines with any existing encoder
    """

    def __init__(
        self,
        model_name: str,
        encoder_type: str = "persoma",
        retriever_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 10,
        index_type: str = "flat",
        # E2P-specific
        num_prefix_tokens: int = 4,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        # PERSOMA-specific
        num_soft_tokens: int = 4,
        adapter_type: str = "perceiver",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            model_name: HuggingFace LLM model
            encoder_type: Underlying encoder ("e2p" or "persoma")
            retriever_encoder: Encoder for FAISS retrieval
            top_k: Number of items to retrieve
            index_type: FAISS index type ("flat" or "ivf")
            num_prefix_tokens: For E2P encoder
            encoder_name: Sentence transformer for E2P
            num_soft_tokens: For PERSOMA encoder
            adapter_type: For PERSOMA encoder ("perceiver" or "mlp")
            device: Compute device
            torch_dtype: Model dtype
        """
        super().__init__(model_name, device, torch_dtype)

        self.encoder_type = encoder_type
        self.retriever_encoder = retriever_encoder
        self.top_k = top_k
        self.index_type = index_type

        # E2P config
        self.num_prefix_tokens = num_prefix_tokens
        self.encoder_name = encoder_name

        # PERSOMA config
        self.num_soft_tokens = num_soft_tokens
        self.adapter_type = adapter_type

        # Components (lazy loaded)
        self.retriever: Optional[FAISSRetriever] = None
        self.base_model: Optional[E2PLLM | PERSOMALLM] = None

        # Current user index cache
        self._current_user_id: Optional[str] = None
        self._current_history: Optional[list[str]] = None

    def load(self) -> None:
        """Load retriever and underlying encoder."""
        # Create retriever
        self.retriever = FAISSRetriever(
            encoder_name=self.retriever_encoder,
            index_type=self.index_type,
            device=self.device,
        )

        # Create base model
        if self.encoder_type == "e2p":
            self.base_model = E2PLLM(
                model_name=self.model_name,
                encoder_name=self.encoder_name,
                num_prefix_tokens=self.num_prefix_tokens,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
        elif self.encoder_type == "persoma":
            self.base_model = PERSOMALLM(
                model_name=self.model_name,
                encoder_name=self.encoder_name,
                num_soft_tokens=self.num_soft_tokens,
                adapter_type=self.adapter_type,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

        self.base_model.load()

        # Expose tokenizer and model from base
        self.tokenizer = self.base_model.tokenizer
        self.model = self.base_model.model

    def _build_user_index(
        self,
        history_items: list[str],
        user_id: Optional[str] = None,
    ) -> None:
        """Build or update user history index."""
        # Check cache
        if user_id and user_id == self._current_user_id:
            if self._current_history == history_items:
                return  # Already indexed

        # Build new index
        self.retriever.build_index(history_items)
        self._current_user_id = user_id
        self._current_history = history_items

    def retrieve_context(
        self,
        query: str,
        history_items: list[str],
        user_id: Optional[str] = None,
    ) -> list[str]:
        """
        Retrieve relevant history items for query.

        Args:
            query: Current query/prompt
            history_items: Full user history
            user_id: Optional user ID for caching

        Returns:
            Top-k relevant history items
        """
        if not history_items:
            return []

        # Build index
        self._build_user_index(history_items, user_id)

        # Retrieve
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        return retrieved

    def generate(
        self,
        prompt: str,
        user_context: Optional[str] = None,
        history_items: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate with retrieval-augmented context.

        If history_items provided:
        1. Retrieve top-k relevant items
        2. Pass to underlying encoder
        3. Generate with soft tokens

        Args:
            prompt: Input prompt
            user_context: Direct user context (passed through if no history)
            history_items: User history items for retrieval
            user_id: Optional user ID for index caching
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generation output
        """
        if history_items:
            # Retrieve relevant items
            retrieved = self.retrieve_context(prompt, history_items, user_id)

            if self.encoder_type == "e2p":
                # E2P expects single context string
                context = " | ".join(retrieved) if retrieved else None
                return self.base_model.generate(
                    prompt,
                    user_context=context,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
            else:
                # PERSOMA expects list of history items
                return self.base_model.generate(
                    prompt,
                    history_items=retrieved,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
        else:
            # Fallback to direct context
            if self.encoder_type == "e2p":
                return self.base_model.generate(
                    prompt,
                    user_context=user_context,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
            else:
                # Convert context to single-item history
                items = [user_context] if user_context else None
                return self.base_model.generate(
                    prompt,
                    history_items=items,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )

    def forward_for_training(
        self,
        prompt: str,
        target: str,
        history_items: list[str],
        use_retrieval: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            prompt: Input prompt
            target: Target output
            history_items: User history items
            use_retrieval: Whether to use retrieval (False uses all items)

        Returns:
            Loss tensor
        """
        if use_retrieval and history_items:
            # Retrieve relevant items
            retrieved = self.retrieve_context(prompt, history_items)
        else:
            # Use all items (or top-k if too many)
            retrieved = history_items[:self.top_k] if history_items else []

        if self.encoder_type == "e2p":
            context = " | ".join(retrieved) if retrieved else ""
            return self.base_model.forward_for_training(prompt, target, context)
        else:
            return self.base_model.forward_for_training(prompt, target, retrieved)

    def get_trainable_parameters(self) -> list:
        """Return trainable parameters from underlying encoder."""
        return self.base_model.get_trainable_parameters()

    def get_num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())

    def save_projector(self, path: str) -> None:
        """Save encoder projector weights."""
        if self.encoder_type == "e2p":
            self.base_model.save_projector(path)
        else:
            self.base_model.save_adapter(path)

    def load_projector(self, path: str) -> None:
        """Load encoder projector weights."""
        if self.encoder_type == "e2p":
            self.base_model.load_projector(path)
        else:
            self.base_model.load_adapter(path)

    def get_retrieval_stats(
        self,
        query: str,
        history_items: list[str],
    ) -> dict:
        """Get statistics about retrieval."""
        if not history_items:
            return {"total_items": 0, "retrieved_items": 0}

        retrieved, scores = self.retriever.retrieve(
            query, top_k=self.top_k, return_scores=True
        )

        return {
            "total_items": len(history_items),
            "retrieved_items": len(retrieved),
            "top_k": self.top_k,
            "scores": scores,
            "retrieved": retrieved,
        }

    def __repr__(self) -> str:
        return (
            f"RetrievalAugmentedLLM(model={self.model_name}, "
            f"encoder={self.encoder_type}, top_k={self.top_k})"
        )
