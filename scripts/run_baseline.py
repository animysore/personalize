#!/usr/bin/env python3
"""Run text baseline personalization."""

import argparse
import json
from pathlib import Path

from personalize.data import UserContext, create_demo_dataset
from personalize.models import TextBaselineLLM


def main():
    parser = argparse.ArgumentParser(description="Run text baseline personalization")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to run (optional, uses demo dataset if not provided)",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="User context as JSON string or path to JSON file",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Maximum tokens for user context",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare outputs with and without context",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    llm = TextBaselineLLM(
        model_name=args.model,
        device=args.device,
        max_context_tokens=args.max_context_tokens,
    )
    llm.load()
    print("Model loaded.\n")

    if args.prompt:
        # Single prompt mode
        user_context = None
        if args.context:
            context_data = _load_context(args.context)
            user_context = UserContext.from_dict(context_data)

        run_single(
            llm,
            args.prompt,
            user_context,
            args.max_new_tokens,
            args.temperature,
            args.compare,
        )
    else:
        # Demo dataset mode
        run_demo(llm, args.max_new_tokens, args.temperature, args.compare)


def _load_context(context_arg: str) -> dict:
    """Load context from JSON string or file path."""
    path = Path(context_arg)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(context_arg)


def run_single(
    llm: TextBaselineLLM,
    prompt: str,
    user_context: UserContext | None,
    max_new_tokens: int,
    temperature: float,
    compare: bool,
):
    """Run a single prompt."""
    context_text = user_context.to_text() if user_context else None

    if compare and context_text:
        print("=" * 60)
        print("WITHOUT CONTEXT:")
        print("=" * 60)
        output_no_ctx = llm.generate(
            prompt,
            user_context=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(f"\nResponse:\n{output_no_ctx.text}")
        print(f"\n[Tokens: {output_no_ctx.prompt_tokens} prompt, {output_no_ctx.generated_tokens} generated]")

        print("\n" + "=" * 60)
        print("WITH CONTEXT:")
        print("=" * 60)

    if context_text:
        print(f"\nUser Context:\n{context_text}\n")
        stats = llm.get_context_stats(context_text)
        print(f"[Context: {stats['final_tokens']} tokens", end="")
        if stats["truncated"]:
            print(f", truncated from {stats['original_tokens']}", end="")
        print("]\n")

    output = llm.generate(
        prompt,
        user_context=context_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print(f"Prompt: {prompt}\n")
    print(f"Response:\n{output.text}")
    print(f"\n[Tokens: {output.prompt_tokens} prompt, {output.generated_tokens} generated]")


def run_demo(
    llm: TextBaselineLLM,
    max_new_tokens: int,
    temperature: float,
    compare: bool,
):
    """Run on demo dataset."""
    dataset = create_demo_dataset()

    for i, sample in enumerate(dataset):
        print("\n" + "=" * 60)
        print(f"SAMPLE {i + 1}")
        print("=" * 60)

        user_context: UserContext = sample["user_context"]
        prompt = sample["prompt"]
        context_text = user_context.to_text()

        if compare and context_text:
            print("\n--- Without Context ---")
            output_no_ctx = llm.generate(
                prompt,
                user_context=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            print(f"Response: {output_no_ctx.text[:200]}...")

            print("\n--- With Context ---")

        if context_text:
            print(f"\nContext:\n{context_text}\n")

        print(f"Prompt: {prompt}\n")

        output = llm.generate(
            prompt,
            user_context=context_text if context_text else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        print(f"Response:\n{output.text}")
        print(f"\n[{output.prompt_tokens} prompt + {output.generated_tokens} generated tokens]")

        if sample["reference"]:
            print(f"\nReference: {sample['reference']}")


if __name__ == "__main__":
    main()
