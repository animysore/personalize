#!/usr/bin/env python3
"""Train Retrieval-Augmented Personalization on LaMP benchmark."""

import argparse

import torch
from torch.optim import AdamW
from tqdm import tqdm

from personalize.datasets import LaMPDataset, LAMP_TASKS
from personalize.models import RetrievalAugmentedLLM


def main():
    parser = argparse.ArgumentParser(description="Train RAP on LaMP")
    parser.add_argument("--task", type=str, default="LaMP-2", choices=list(LAMP_TASKS.keys()))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--encoder-type", type=str, default="persoma", choices=["e2p", "persoma"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-soft-tokens", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="./data/lamp")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = f"rap_{args.task.lower()}_{args.encoder_type}.pt"

    print("=" * 70)
    print(f"Training RAP on {args.task}: {LAMP_TASKS[args.task]['name']}")
    print("=" * 70)
    print(f"Encoder type: {args.encoder_type}")
    print(f"Top-k retrieval: {args.top_k}")

    # Load datasets
    print(f"\nLoading {args.task} dataset...")
    train_data = LaMPDataset(
        args.task,
        split="train",
        data_dir=args.data_dir,
        max_profile_items=50,  # Full history for retrieval
    )
    print(f"Train samples: {len(train_data)}")

    dev_data = LaMPDataset(
        args.task,
        split="dev",
        data_dir=args.data_dir,
        max_profile_items=50,
    )
    print(f"Dev samples: {len(dev_data)}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model = RetrievalAugmentedLLM(
        model_name=args.model,
        encoder_type=args.encoder_type,
        top_k=args.top_k,
        num_soft_tokens=args.num_soft_tokens,
        device=args.device,
    )
    model.load()
    print(f"Trainable parameters: {model.get_num_trainable_parameters():,}")

    # Setup optimizer
    optimizer = AdamW(model.get_trainable_parameters(), lr=args.lr)

    # Training
    print(f"\nTraining for {args.epochs} epochs on {min(len(train_data), args.max_samples)} samples...")
    print("-" * 70)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_samples = 0

        samples = list(train_data)[:args.max_samples]
        progress = tqdm(samples, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for sample in progress:
            if sample.output is None:
                continue

            # Get profile items as history
            history_items = []
            if sample.profile:
                for item in sample.profile[:50]:
                    if isinstance(item, dict):
                        # LaMP-2 has 'tag' and 'description' keys
                        tag = item.get("tag", "")
                        desc = item.get("description", "") or item.get("text", "")
                        if tag and desc:
                            history_items.append(f"{tag}: {desc[:200]}")
                        elif desc:
                            history_items.append(desc[:200])

            if not history_items:
                continue

            formatted = train_data.format_for_training(sample)
            prompt = formatted["prompt"]
            target = formatted["target"]

            if not target:
                continue

            optimizer.zero_grad()

            try:
                loss = model.forward_for_training(
                    prompt, target, history_items, use_retrieval=True
                )

                if loss.item() == 0.0 or torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_samples += 1

                if num_samples % 50 == 0:
                    progress.set_postfix({"loss": f"{epoch_loss / num_samples:.4f}"})

            except Exception:
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_samples if num_samples > 0 else 0
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f} ({num_samples} samples)")

        # Evaluation
        eval_loss = evaluate(model, dev_data, args.eval_samples)
        print(f"  Dev loss: {eval_loss:.4f}")

    print("-" * 70)
    print("Training complete!")

    # Save
    print(f"\nSaving to {args.save_path}...")
    model.save_projector(args.save_path)

    # Test generation
    print("\n" + "=" * 70)
    print("Sample Outputs")
    print("=" * 70)

    for i, sample in enumerate(dev_data):
        if i >= 3:
            break
        if sample.output is None:
            continue

        history_items = []
        if sample.profile:
            for item in sample.profile[:50]:
                if isinstance(item, dict):
                    tag = item.get("tag", "")
                    desc = item.get("description", "") or item.get("text", "")
                    if tag and desc:
                        history_items.append(f"{tag}: {desc[:200]}")
                    elif desc:
                        history_items.append(desc[:200])

        print(f"\n--- Sample {i + 1} ---")
        print(f"Prompt: {sample.input[:100]}...")
        print(f"History items: {len(history_items)}")
        print(f"Target: {sample.output[:100]}...")

        # Show retrieval stats
        stats = model.get_retrieval_stats(sample.input, history_items)
        print(f"Retrieved {stats['retrieved_items']} items from {stats['total_items']}")

        output = model.generate(
            sample.input,
            history_items=history_items,
            max_new_tokens=50,
            temperature=0.7,
        )
        print(f"RAP Output: {output.text[:100]}...")


@torch.no_grad()
def evaluate(model: RetrievalAugmentedLLM, dataset: LaMPDataset, max_samples: int) -> float:
    """Evaluate model on dataset."""
    total_loss = 0.0
    num_samples = 0

    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        if sample.output is None:
            continue

        history_items = []
        if sample.profile:
            for item in sample.profile[:50]:
                if isinstance(item, dict):
                    tag = item.get("tag", "")
                    desc = item.get("description", "") or item.get("text", "")
                    if tag and desc:
                        history_items.append(f"{tag}: {desc[:200]}")
                    elif desc:
                        history_items.append(desc[:200])

        if not history_items:
            continue

        formatted = dataset.format_for_training(sample)
        prompt = formatted["prompt"]
        target = formatted["target"]

        if not target:
            continue

        try:
            loss = model.forward_for_training(prompt, target, history_items)
            total_loss += loss.item()
            num_samples += 1
        except Exception:
            continue

    return total_loss / num_samples if num_samples > 0 else 0.0


if __name__ == "__main__":
    main()
