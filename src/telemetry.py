from __future__ import annotations

import contextlib
import logging
import math
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from .config import LoggingConfig
from .utils import gpu_memory_stats, human_bytes, jsonl_append

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def build_logger(name: str, level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class Telemetry:
    def __init__(self, root_dir: Path, logging_cfg: LoggingConfig) -> None:
        self.root_dir = root_dir
        self.logging_cfg = logging_cfg
        self.jsonl_path = root_dir / "logs" / "jsonl" / "metrics.jsonl"
        self.writer = SummaryWriter(str(root_dir / "logs" / "tensorboard")) if logging_cfg.tensorboard and SummaryWriter else None

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        if self.logging_cfg.jsonl:
            jsonl_append(self.jsonl_path, {"step": step, **metrics})

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class ThroughputMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.positions = 0
        self.nodes = 0
        self.start_time = time.perf_counter()

    def update(self, positions: int, nodes: int) -> None:
        self.positions += positions
        self.nodes += nodes

    def summary(self) -> Dict[str, float]:
        elapsed = max(time.perf_counter() - self.start_time, 1e-8)
        return {
            "positions_per_sec": self.positions / elapsed,
            "nodes_per_sec": self.nodes / elapsed,
            "throughput_elapsed_s": elapsed,
        }


def policy_entropy_from_logits(logits: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if legal_mask is not None:
        logits = logits.masked_fill(~legal_mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-9))
    return -(probs * log_probs).sum(dim=-1).mean()


def estimate_tflops_from_profile(
    step_fn: Callable[[], torch.Tensor],
    device: torch.device,
) -> float:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
        loss = step_fn()
        if isinstance(loss, torch.Tensor):
            loss.backward()
    total_flops = 0.0
    for event in prof.key_averages():
        total_flops += float(getattr(event, "flops", 0.0))
    return total_flops / 1.0e12


def projected_elo(score_fraction: float, opponent_elo: float) -> float:
    score_fraction = min(max(score_fraction, 1e-4), 1.0 - 1e-4)
    delta = -400.0 * math.log10((1.0 / score_fraction) - 1.0)
    return opponent_elo + delta


def device_metrics(device: torch.device) -> Dict[str, float]:
    allocated, reserved = gpu_memory_stats(device)
    return {
        "gpu_mem_allocated_mb": allocated / (1024.0 ** 2),
        "gpu_mem_reserved_mb": reserved / (1024.0 ** 2),
    }