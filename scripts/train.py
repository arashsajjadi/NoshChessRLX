from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.trainer import HybridTrainer


def detect_checkpoint_kind(checkpoint: object) -> str:
    if not isinstance(checkpoint, dict):
        return "unknown"
    if "model_state_dict" not in checkpoint:
        return "unknown"
    if "optimizer_state_dict" in checkpoint:
        return "full"
    return "weights_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid chess RL trainer")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to YAML config.")
    parser.add_argument("--stage", type=str, choices=["phase1", "phase2", "full"], default=None, help="Override run stage.")
    parser.add_argument("--continue", dest="continue_from", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--clear_cache", action="store_true", help="Clear cache before the run.")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default=None, help="Override device.")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default=None, help="Override precision.")
    parser.add_argument("--verbosity", type=str, choices=["silent", "info", "debug"], default=None, help="Override log verbosity.")
    parser.add_argument("--profile", action="store_true", help="Enable profiler sampling.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.stage is not None:
        config.run.stage = args.stage
    if args.continue_from is not None:
        config.run.continue_from = args.continue_from
    if args.clear_cache:
        config.run.clear_cache = True
    if args.device is not None:
        config.project.device = args.device
    if args.dtype is not None:
        config.project.dtype = args.dtype
    if args.verbosity is not None:
        config.project.verbosity = args.verbosity
    if args.profile:
        config.run.profile = True
    if args.no_compile:
        config.project.compile = False

    trainer = HybridTrainer(config)
    if config.run.continue_from:
        checkpoint_path = Path(config.run.continue_from).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # PyTorch 2.6 defaults torch.load to weights_only=True.
        # Our trusted local checkpoints may include optimizer state and numpy objects.
        # Use weights_only=False for local resume/initialization payloads.
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_kind = detect_checkpoint_kind(checkpoint_payload)
        if checkpoint_kind == "full":
            trainer.load_checkpoint(checkpoint_path, checkpoint=checkpoint_payload)
        elif checkpoint_kind == "weights_only":
            trainer.load_model_weights_only(checkpoint_path, checkpoint=checkpoint_payload)
        else:
            raise RuntimeError(
                f"Unsupported checkpoint format at {checkpoint_path}. "
                "Expected full checkpoint or weight-only payload with model_state_dict."
            )
    trainer.run()


if __name__ == "__main__":
    main()
