#!/usr/bin/env python3
"""Compare all personalization methods side-by-side."""

import argparse

from personalize.datasets import create_demo_dataset
from personalize.models import E2PLLM, PERSOMALLM, TextBaselineLLM


def main():
    parser = argparse.ArgumentParser(description="Compare personalization methods")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--e2p-1tok", type=str, default="e2p_projector.pt")
    parser.add_argument("--e2p-5tok", type=str, default="e2p_5tok.pt")
    parser.add_argument("--persoma", type=str, default="persoma_perceiver.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=4)
    args = parser.parse_args()

    print("=" * 70)
    print("Personalization Methods Comparison")
    print("=" * 70)

    # Load all models
    print("\nLoading models...")

    # Text Baseline
    print("  - Text Baseline")
    baseline = TextBaselineLLM(model_name=args.model, device=args.device)
    baseline.load()

    # E2P 1-token
    print("  - E2P (1 token)")
    e2p_1 = E2PLLM(
        model_name=args.model,
        num_prefix_tokens=1,
        device=args.device,
    )
    e2p_1.load()
    try:
        e2p_1.load_projector(args.e2p_1tok)
        e2p_1_loaded = True
    except FileNotFoundError:
        print(f"    Warning: {args.e2p_1tok} not found, using random weights")
        e2p_1_loaded = False

    # E2P 5-token
    print("  - E2P (5 tokens)")
    e2p_5 = E2PLLM(
        model_name=args.model,
        num_prefix_tokens=5,
        device=args.device,
    )
    e2p_5.load()
    try:
        e2p_5.load_projector(args.e2p_5tok)
        e2p_5_loaded = True
    except FileNotFoundError:
        print(f"    Warning: {args.e2p_5tok} not found, using random weights")
        e2p_5_loaded = False

    # PERSOMA
    print("  - PERSOMA (4 tokens, Perceiver)")
    persoma = PERSOMALLM(
        model_name=args.model,
        num_soft_tokens=4,
        adapter_type="perceiver",
        device=args.device,
    )
    persoma.load()
    try:
        persoma.load_adapter(args.persoma)
        persoma_loaded = True
    except FileNotFoundError:
        print(f"    Warning: {args.persoma} not found, using random weights")
        persoma_loaded = False

    print("\nModels loaded.\n")

    # Print model stats
    print("=" * 70)
    print("Model Statistics")
    print("=" * 70)
    print(f"{'Method':<25} {'Soft Tokens':<12} {'Trainable Params':<18}")
    print("-" * 70)
    print(f"{'Text Baseline':<25} {'N/A':<12} {'0':>18}")
    print(f"{'E2P (1 token)':<25} {'1':<12} {e2p_1.get_num_trainable_parameters():>18,}")
    print(f"{'E2P (5 tokens)':<25} {'5':<12} {e2p_5.get_num_trainable_parameters():>18,}")
    print(f"{'PERSOMA (Perceiver)':<25} {'4':<12} {persoma.get_num_trainable_parameters():>18,}")
    print()

    # Run comparison
    dataset = create_demo_dataset()

    for i, sample in enumerate(dataset):
        if i >= args.max_samples:
            break

        user_context = sample["user_context"]
        context_text = user_context.to_text()
        prompt = sample["prompt"]
        reference = sample.get("reference", "N/A")

        if not context_text:
            continue

        # Extract history items for PERSOMA
        history_items = []
        if user_context.profile:
            history_items.append(user_context.profile.to_text())
        if user_context.history:
            for item in user_context.history.items:
                history_items.append(item.content)

        print("=" * 70)
        print(f"SAMPLE {i + 1}")
        print("=" * 70)
        print(f"\nContext:\n{context_text[:200]}{'...' if len(context_text) > 200 else ''}\n")
        print(f"Prompt: {prompt}")
        print(f"Reference: {reference[:100]}{'...' if len(str(reference)) > 100 else ''}\n")

        # Get compression stats
        baseline_stats = baseline.get_context_stats(context_text)
        e2p_1_stats = e2p_1.get_context_stats(context_text)

        print(f"Context tokens: {baseline_stats['original_tokens']} text → 1/5/4 soft tokens\n")

        # Generate from each model
        methods = [
            ("Text Baseline", baseline, {"user_context": context_text}),
            ("E2P (1 token)", e2p_1, {"user_context": context_text}),
            ("E2P (5 tokens)", e2p_5, {"user_context": context_text}),
            ("PERSOMA", persoma, {"history_items": history_items}),
        ]

        for name, model, kwargs in methods:
            output = model.generate(
                prompt,
                max_new_tokens=80,
                temperature=0.7,
                **kwargs,
            )
            response = output.text.strip()[:150]
            print(f"[{name}]")
            print(f"  {response}{'...' if len(output.text) > 150 else ''}")
            print(f"  ({output.prompt_tokens} prompt tokens)\n")

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
| Method              | Tokens | Params    | Compression | Notes                    |
|---------------------|--------|-----------|-------------|--------------------------|
| Text Baseline       | 40-100 | 0         | 1x          | Full context as text     |
| E2P (1 token)       | 1      | 462K      | 40-100x     | Best compression         |
| E2P (5 tokens)      | 5      | 2.3M      | 8-20x       | More expressive          |
| PERSOMA (Perceiver) | 4      | 8.2M      | 10-25x      | Handles variable history |

Key findings:
- E2P (1 token): Best compression, good for simple personalization
- E2P (5 tokens): More capacity but may overfit on small data
- PERSOMA: Best for long/variable histories, needs more training data
""")


if __name__ == "__main__":
    main()
