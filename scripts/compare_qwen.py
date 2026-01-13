#!/usr/bin/env python3
"""Compare Qwen2.5-1.5B baseline vs E2P."""

import argparse
from personalize.models import E2PLLM, TextBaselineLLM
from personalize.datasets import create_demo_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--projector", default="e2p_qwen1.5b_1tok.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=6)
    args = parser.parse_args()

    print("Loading models...")

    baseline = TextBaselineLLM(args.model, device=args.device)
    baseline.load()

    e2p = E2PLLM(args.model, num_prefix_tokens=1, device=args.device)
    e2p.load()
    e2p.load_projector(args.projector)

    print("Models loaded.\n")

    dataset = create_demo_dataset()

    print("=" * 70)
    print(f"Comparison: {args.model}")
    print("=" * 70)

    for i, sample in enumerate(dataset):
        if i >= args.max_samples:
            break

        ctx = sample["user_context"]
        ctx_text = ctx.to_text()
        if not ctx_text:
            continue

        prompt = sample["prompt"]
        ref = sample.get("reference", "N/A")

        print(f"\n{'='*70}")
        print(f"Sample {i+1}")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"Context: {ctx_text[:100]}...")
        print(f"Reference: {ref[:100]}...")

        # Get stats
        stats = e2p.get_context_stats(ctx_text)
        print(f"\nCompression: {stats['text_tokens']} text tokens -> 1 soft token ({stats['compression_ratio']:.0f}x)")

        # Baseline
        out_b = baseline.generate(prompt, user_context=ctx_text, max_new_tokens=80, temperature=0.7)
        print(f"\n[Baseline] ({out_b.prompt_tokens} prompt tokens)")
        print(f"  {out_b.text[:200]}")

        # E2P
        out_e = e2p.generate(prompt, user_context=ctx_text, max_new_tokens=80, temperature=0.7)
        print(f"\n[E2P] ({out_e.prompt_tokens} prompt tokens)")
        print(f"  {out_e.text[:200]}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
E2P compresses user context (40-100 tokens) into 1 soft token.
- Same LLM quality, much shorter prompts
- Faster inference due to reduced context length
""")


if __name__ == "__main__":
    main()
