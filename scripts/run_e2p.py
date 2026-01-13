#!/usr/bin/env python3
"""Run E2P (Embedding-to-Prefix) personalization."""

import argparse

from personalize.data import create_demo_dataset
from personalize.models import E2PLLM, TextBaselineLLM


def main():
    parser = argparse.ArgumentParser(description="Run E2P personalization")
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="HuggingFace LLM model name",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer encoder model",
    )
    parser.add_argument(
        "--num-prefix-tokens",
        type=int,
        default=1,
        help="Number of soft prefix tokens",
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
        "--compare-baseline",
        action="store_true",
        help="Compare with text baseline",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("E2P (Embedding-to-Prefix) Personalization Demo")
    print("=" * 60)

    # Load E2P model
    print(f"\nLoading E2P model: {args.model}")
    print(f"User encoder: {args.encoder}")
    print(f"Prefix tokens: {args.num_prefix_tokens}")

    e2p = E2PLLM(
        model_name=args.model,
        encoder_name=args.encoder,
        num_prefix_tokens=args.num_prefix_tokens,
        device=args.device,
    )
    e2p.load()

    print(f"\nE2P projector parameters: {e2p.get_num_trainable_parameters():,}")
    print("Model loaded.\n")

    # Optionally load baseline for comparison
    baseline = None
    if args.compare_baseline:
        print("Loading text baseline for comparison...")
        baseline = TextBaselineLLM(
            model_name=args.model,
            device=args.device,
        )
        baseline.load()
        print("Baseline loaded.\n")

    # Run on demo dataset
    dataset = create_demo_dataset()

    for i, sample in enumerate(dataset):
        print("\n" + "=" * 60)
        print(f"SAMPLE {i + 1}")
        print("=" * 60)

        user_context = sample["user_context"]
        prompt = sample["prompt"]
        context_text = user_context.to_text()

        if not context_text:
            print(f"\nPrompt: {prompt}")
            print("(No user context - skipping personalization)")
            continue

        print(f"\nUser Context:\n{context_text}\n")

        # Show compression stats
        stats = e2p.get_context_stats(context_text)
        print(f"[Context Encoding Stats]")
        print(f"  Text tokens: {stats['text_tokens']}")
        print(f"  Soft tokens: {stats['soft_tokens']}")
        print(f"  Compression: {stats['compression_ratio']:.1f}x")
        print(f"  User embedding dim: {stats['user_embedding_dim']}")

        print(f"\nPrompt: {prompt}\n")

        # Compare with baseline if requested
        if baseline and args.compare_baseline:
            print("--- Text Baseline ---")
            baseline_output = baseline.generate(
                prompt,
                user_context=context_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print(f"Response: {baseline_output.text[:300]}...")
            print(f"[{baseline_output.prompt_tokens} prompt tokens]")

            print("\n--- E2P (Soft Prefix) ---")

        # E2P generation
        e2p_output = e2p.generate(
            prompt,
            user_context=context_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print(f"Response:\n{e2p_output.text}")
        print(f"\n[{e2p_output.prompt_tokens} effective tokens ({args.num_prefix_tokens} soft + {e2p_output.prompt_tokens - args.num_prefix_tokens} text)]")

        if sample["reference"]:
            print(f"\nReference: {sample['reference']}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: E2P vs Text Baseline")
    print("=" * 60)
    print(f"""
E2P compresses user context into {args.num_prefix_tokens} soft token(s) instead of
injecting raw text (which can be 50-500+ tokens).

Benefits:
- Reduced prompt length -> faster inference
- Fixed context size regardless of history length
- Learnable representation can capture user patterns

Note: The projector is randomly initialized in this demo.
For best results, train it on personalization data.
""")


if __name__ == "__main__":
    main()
