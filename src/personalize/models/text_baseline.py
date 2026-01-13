"""Text baseline: User context injected as plain text prefix."""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import GenerationOutput, PersonalizedLLM


class TextBaselineLLM(PersonalizedLLM):
    """
    Baseline personalization via raw text context injection.

    Simply prepends user context as text to the prompt. This serves as a
    baseline to compare against more efficient methods like soft prompts
    or encoder-based compression.

    Example:
        >>> llm = TextBaselineLLM("meta-llama/Llama-3.2-1B")
        >>> llm.load()
        >>> context = "User preferences: likes sci-fi, dislikes horror"
        >>> output = llm.generate("Recommend a movie", user_context=context)
    """

    DEFAULT_CONTEXT_TEMPLATE = """### User Context
{context}

### Instruction
{prompt}

### Response
"""

    DEFAULT_NO_CONTEXT_TEMPLATE = """### Instruction
{prompt}

### Response
"""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        context_template: Optional[str] = None,
        no_context_template: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
    ):
        """
        Initialize text baseline model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            torch_dtype: Model dtype (default: float16)
            context_template: Template for prompt with context.
                Must contain {context} and {prompt} placeholders.
            no_context_template: Template for prompt without context.
                Must contain {prompt} placeholder.
            max_context_tokens: Maximum tokens for user context (truncates if exceeded)
        """
        super().__init__(model_name, device, torch_dtype)
        self.context_template = context_template or self.DEFAULT_CONTEXT_TEMPLATE
        self.no_context_template = no_context_template or self.DEFAULT_NO_CONTEXT_TEMPLATE
        self.max_context_tokens = max_context_tokens

    def load(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def _truncate_context(self, context: str) -> str:
        """Truncate context to max_context_tokens if specified."""
        if self.max_context_tokens is None:
            return context

        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) <= self.max_context_tokens:
            return context

        truncated_tokens = tokens[: self.max_context_tokens]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    def _build_prompt(self, prompt: str, user_context: Optional[str] = None) -> str:
        """Build full prompt with optional user context."""
        if user_context:
            context = self._truncate_context(user_context)
            return self.context_template.format(context=context, prompt=prompt)
        return self.no_context_template.format(prompt=prompt)

    def generate(
        self,
        prompt: str,
        user_context: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate personalized response.

        Args:
            prompt: User's input prompt/question
            user_context: Optional user context (preferences, history, etc.)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (False = greedy)
            **kwargs: Additional generation arguments

        Returns:
            GenerationOutput with generated text and token counts
        """
        full_prompt = self._build_prompt(prompt, user_context)

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings - max_new_tokens,
        ).to(self.device)

        prompt_tokens = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        generated_tokens = outputs.shape[1] - prompt_tokens
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response (after the prompt)
        response = self.tokenizer.decode(
            outputs[0][prompt_tokens:], skip_special_tokens=True
        ).strip()

        return GenerationOutput(
            text=response,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_tokens=outputs.shape[1],
        )

    def get_context_stats(self, user_context: str) -> dict:
        """Get statistics about user context."""
        tokens = self.tokenizer.encode(user_context, add_special_tokens=False)
        return {
            "original_tokens": len(tokens),
            "truncated": self.max_context_tokens is not None
            and len(tokens) > self.max_context_tokens,
            "final_tokens": min(len(tokens), self.max_context_tokens or len(tokens)),
        }
