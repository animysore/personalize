#!/usr/bin/env python3
"""Evaluate E2P on LaMP benchmark tasks."""

import argparse
from collections import Counter

import torch
from tqdm import tqdm

from personalize.datasets import LaMPDataset, LAMP_TASKS
from personalize.models import E2PLLM, TextBaselineLLM


def accuracy(predictions: list[str], targets: list[str]) -> float:
    """Compute exact match accuracy."""
    correct = sum(1 for p, t in zip(predictions, targets) if p.strip().lower() == t.strip().lower())
    return correct / len(targets) if targets else 0.0


def tag_accuracy(predictions: list[str], targets: list[str], valid_tags: list[str]) -> float:
    """Compute accuracy for tag prediction, extracting tags from predictions."""
    correct = 0
    for pred, target in zip(predictions, targets):
        # Try to extract a valid tag from the prediction
        pred_lower = pred.lower()
        target_lower = target.strip().lower()

        # Check if target tag appears in prediction
        if target_lower in pred_lower:
            correct += 1
        else:
            # Check if any valid tag appears
            for tag in valid_tags:
                if tag.lower() in pred_lower:
                    if tag.lower() == target_lower:
                        correct += 1
                    break

    return correct / len(targets) if targets else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate E2P on LaMP")
    parser.add_argument("--task", type=str, default="LaMP-2", choices=list(LAMP_TASKS.keys()))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--projector", type=str, default="e2p_lamp-2_1tok.pt")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-profile-items", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="./data/lamp")
    parser.add_argument("--compare-baseline", action="store_true", help="Also evaluate text baseline")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Evaluating on {args.task}: {LAMP_TASKS[args.task]['name']}")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading {args.task} dev set...")
    dev_data = LaMPDataset(
        args.task,
        split="dev",
        data_dir=args.data_dir,
        max_profile_items=args.max_profile_items,
    )
    print(f"Dev samples: {len(dev_data)}")

    # Load E2P model
    print(f"\nLoading E2P model: {args.model}")
    e2p = E2PLLM(
        model_name=args.model,
        num_prefix_tokens=1,
        device=args.device,
    )
    e2p.load()
    e2p.load_projector(args.projector)
    e2p.prefix_projector.eval()

    # Load baseline if requested
    baseline = None
    if args.compare_baseline:
        print("Loading text baseline...")
        baseline = TextBaselineLLM(args.model, device=args.device)
        baseline.load()

    # Get valid tags for the task (for LaMP-2)
    valid_tags = [
        "sci-fi", "based on a book", "comedy", "action", "twist ending",
        "dystopia", "dark comedy", "classic", "psychology", "fantasy",
        "romance", "thought-provoking", "social commentary", "violence", "true story"
    ]

    # Evaluate
    print(f"\nEvaluating on {min(len(dev_data), args.max_samples)} samples...")
    print("-" * 70)

    e2p_preds = []
    baseline_preds = []
    targets = []

    for i, sample in enumerate(tqdm(dev_data, total=min(len(dev_data), args.max_samples))):
        if i >= args.max_samples:
            break
        if sample.output is None:
            continue

        formatted = dev_data.format_for_training(sample)
        context_text = formatted["context"]
        prompt = formatted["prompt"]
        target = formatted["target"]

        if not context_text or not target:
            continue

        targets.append(target)

        # E2P prediction
        try:
            with torch.no_grad():
                output = e2p.generate(
                    prompt,
                    user_context=context_text,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=False,
                )
            e2p_preds.append(output.text)
        except Exception:
            e2p_preds.append("")

        # Baseline prediction
        if baseline:
            try:
                with torch.no_grad():
                    output = baseline.generate(
                        prompt,
                        user_context=context_text,
                        max_new_tokens=20,
                        temperature=0.1,
                        do_sample=False,
                    )
                baseline_preds.append(output.text)
            except Exception:
                baseline_preds.append("")

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute metrics
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    e2p_acc = accuracy(e2p_preds, targets)
    e2p_tag_acc = tag_accuracy(e2p_preds, targets, valid_tags)

    print(f"\nE2P Model:")
    print(f"  Exact Match Accuracy: {e2p_acc:.2%}")
    print(f"  Tag Extraction Accuracy: {e2p_tag_acc:.2%}")

    if baseline:
        baseline_acc = accuracy(baseline_preds, targets)
        baseline_tag_acc = tag_accuracy(baseline_preds, targets, valid_tags)
        print(f"\nText Baseline:")
        print(f"  Exact Match Accuracy: {baseline_acc:.2%}")
        print(f"  Tag Extraction Accuracy: {baseline_tag_acc:.2%}")

    # Show some examples
    print("\n" + "=" * 70)
    print("Sample Predictions")
    print("=" * 70)

    for i in range(min(5, len(targets))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Target: {targets[i]}")
        print(f"E2P: {e2p_preds[i][:100]}")
        if baseline:
            print(f"Baseline: {baseline_preds[i][:100]}")

    # Distribution of predictions
    print("\n" + "=" * 70)
    print("Prediction Distribution")
    print("=" * 70)

    # Extract tags from E2P predictions
    e2p_tags = []
    for pred in e2p_preds:
        pred_lower = pred.lower()
        found = False
        for tag in valid_tags:
            if tag.lower() in pred_lower:
                e2p_tags.append(tag)
                found = True
                break
        if not found:
            e2p_tags.append("other")

    tag_dist = Counter(e2p_tags)
    print("\nE2P predicted tags:")
    for tag, count in tag_dist.most_common(10):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
