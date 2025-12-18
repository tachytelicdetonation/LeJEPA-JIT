#!/usr/bin/env python
"""
Quick verification script for testing dashboard and training code changes.

This script runs a minimal training session (~2-3 minutes) to verify:
1. All imports work correctly (no missing modules)
2. Training loop runs without errors
3. All diagnostics and visualizations generate
4. WandB logging works correctly

Usage:
    python verify.py                    # Run with WandB enabled
    python verify.py --no_wandb         # Run without WandB
    python verify.py --samples 50       # Use only 50 training samples
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Quick verification run for LeJEPA-JIT"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of training samples to use (default: 100)",
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=50,
        help="Number of validation samples to use (default: 50)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="jit",
        choices=["jit", "vit"],
        help="Encoder type (default: jit)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VERIFICATION RUN")
    print("=" * 60)
    print(f"  Encoder:         {args.encoder}")
    print(f"  Train samples:   {args.samples}")
    print(f"  Val samples:     {args.val_samples}")
    print(f"  WandB:           {'disabled' if args.no_wandb else 'enabled'}")
    print("=" * 60)
    print()

    # Build train.py CLI args
    train_args = [
        "train.py",
        f"--encoder={args.encoder}",
        "--epochs=1",
        "--batch_size=2",
        "--num_workers=0",
        f"--max_train_samples={args.samples}",
        f"--max_val_samples={args.val_samples}",
    ]

    if args.no_wandb:
        train_args.append("--no_wandb")

    # Patch sys.argv and run train.main()
    original_argv = sys.argv
    sys.argv = train_args

    try:
        from train import main as train_main

        train_main()

        print()
        print("=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print("VERIFICATION FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
