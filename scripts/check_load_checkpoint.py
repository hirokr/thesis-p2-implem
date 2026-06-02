#!/usr/bin/env python3
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Check that a PyTorch checkpoint can be loaded (CPU).")
    parser.add_argument("path", help="Path to the checkpoint file")
    parser.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="Allow non-weights-only loading (executes pickled code). Use only with trusted checkpoints.",
    )
    args = parser.parse_args()

    try:
        import torch
    except Exception as e:
        print("ERROR: PyTorch not installed. Install with: pip install torch", file=sys.stderr)
        return 2

    p = args.path
    try:
        ckpt = torch.load(
            p,
            map_location=torch.device("cpu"),
            weights_only=not args.allow_unsafe,
        )
    except Exception as e:
        print(f"FAILED to load checkpoint: {e}", file=sys.stderr)
        if not args.allow_unsafe:
            print(
                "Hint: re-run with --allow-unsafe if you trust the checkpoint source.",
                file=sys.stderr,
            )
        return 1

    if isinstance(ckpt, dict):
        print("Loaded checkpoint. Top-level keys:")
        for k in ckpt.keys():
            print(f" - {k}")
    else:
        print(f"Loaded checkpoint object of type {type(ckpt)}")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
