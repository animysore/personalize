#!/usr/bin/env python3
"""Train PERSOMA adapter on demo data."""

import argparse

import torch
from torch.optim import AdamW

from personalize.datasets import create_demo_dataset
from personalize.models import PERSOMALLM


def main():
    parser = argparse.ArgumentParser(description="Train PERSOMA adapter")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--num-soft-tokens", type=int, default=4)
    parser.add_argument("--adapter-type", type=str, default="perceiver", choices=["perceiver", "mlp"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="persoma_adapter.pt")
    args = parser.parse_args()

    print("=" * 60)
    print("PERSOMA Adapter Training")
    print("=" * 60)

    print(f"\nLoading model: {args.model}")
    print(f"Adapter type: {args.adapter_type}")
    print(f"Soft tokens: {args.num_soft_tokens}")

    model = PERSOMALLM(
        model_name=args.model,
        encoder_name=args.encoder,
        num_soft_tokens=args.num_soft_tokens,
        adapter_type=args.adapter_type,
        device=args.device,
    )
    model.load()

    print(f"Trainable parameters: {model.get_num_trainable_parameters():,}")

    dataset = create_demo_dataset()
    print(f"Training samples: {len(dataset)}")

    optimizer = AdamW(model.get_trainable_parameters(), lr=args.lr)

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    model.persoma.adapter.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_samples = 0

        for sample in dataset:
            user_context = sample["user_context"]

            # Extract history items
            history_items = []
            if user_context.profile:
                history_items.append(user_context.profile.to_text())
            if user_context.history:
                for item in user_context.history.items:
                    history_items.append(item.content)

            if not history_items or not sample["reference"]:
                continue

            prompt = sample["prompt"]
            target = sample["reference"]

            optimizer.zero_grad()

            loss = model.forward_for_training(prompt, target, history_items)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_samples += 1

        avg_loss = epoch_loss / num_samples if num_samples > 0 else 0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}")

    print("-" * 60)
    print("Training complete!\n")

    print(f"Saving adapter to {args.save_path}...")
    model.save_adapter(args.save_path)
    print("Saved.")

    # Test
    print("\n" + "=" * 60)
    print("Testing trained model")
    print("=" * 60)

    model.persoma.adapter.eval()

    test_cases = [
        {
            "prompt": "Recommend a movie for me.",
            "history": ["Preferences: sci-fi, thriller", "Loved Inception", "Loved The Matrix"],
        },
        {
            "prompt": "Write a greeting.",
            "history": ["Style: casual and friendly", "Tone: warm"],
        },
        {
            "prompt": "Explain variables.",
            "history": ["Level: beginner", "No coding experience"],
        },
    ]

    for i, tc in enumerate(test_cases):
        print(f"\n--- Test {i + 1} ---")
        print(f"History items: {tc['history']}")
        print(f"Prompt: {tc['prompt']}")

        output = model.generate(
            tc["prompt"],
            history_items=tc["history"],
            max_new_tokens=100,
            temperature=0.7,
        )
        print(f"Response: {output.text[:200]}")


if __name__ == "__main__":
    main()
