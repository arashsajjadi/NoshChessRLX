from __future__ import annotations

import copy
import math
import platform
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import chess
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .buffer import RingBuffer, SelfPlayDataset, SelfPlayTransition, TeacherDataset, TeacherSample
from .config import TrainConfig
from .encoding import BoardEncoder, MoveEncoder
from .env import RewardShaper
from .mcts import MCTS
from .model import PolicyValueNet
from .teacher import StockfishTeacher, TeacherAnalysis
from .telemetry import (
    Telemetry,
    ThroughputMeter,
    build_logger,
    device_metrics,
    estimate_tflops_from_profile,
    policy_entropy_from_logits,
    projected_elo,
)
from .utils import (
    atomic_torch_save,
    clear_cache_dirs,
    configure_torch_runtime,
    count_parameters,
    ensure_dirs,
    format_seconds,
    get_rng_state,
    maybe_compile,
    resolve_device,
    set_rng_state,
    set_seed,
    to_device,
)


@dataclass(slots=True)
class AdaptiveGAE:
    lambda_min: float
    lambda_max: float
    current_lambda: float
    ema_beta: float
    k_noise: float
    k_value_var: float
    td_var_ema: float = 0.0
    value_var_ema: float = 0.0

    def update(self, values: np.ndarray, td_errors: np.ndarray) -> float:
        value_var = float(np.var(values))
        td_var = float(np.var(td_errors))
        self.value_var_ema = self.ema_beta * self.value_var_ema + (1.0 - self.ema_beta) * value_var
        self.td_var_ema = self.ema_beta * self.td_var_ema + (1.0 - self.ema_beta) * td_var
        signal = self.k_value_var * math.log1p(self.value_var_ema) - self.k_noise * math.log1p(self.td_var_ema)
        sigma = 1.0 / (1.0 + math.exp(-signal))
        raw_lambda = self.lambda_min + (self.lambda_max - self.lambda_min) * sigma
        self.current_lambda = self.ema_beta * self.current_lambda + (1.0 - self.ema_beta) * raw_lambda
        self.current_lambda = float(min(max(self.current_lambda, self.lambda_min), self.lambda_max))
        return self.current_lambda

    def state_dict(self) -> Dict[str, float]:
        return {
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "current_lambda": self.current_lambda,
            "ema_beta": self.ema_beta,
            "k_noise": self.k_noise,
            "k_value_var": self.k_value_var,
            "td_var_ema": self.td_var_ema,
            "value_var_ema": self.value_var_ema,
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.lambda_min = float(state["lambda_min"])
        self.lambda_max = float(state["lambda_max"])
        self.current_lambda = float(state["current_lambda"])
        self.ema_beta = float(state["ema_beta"])
        self.k_noise = float(state["k_noise"])
        self.k_value_var = float(state["k_value_var"])
        self.td_var_ema = float(state["td_var_ema"])
        self.value_var_ema = float(state["value_var_ema"])


class HybridTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.root_dir = config.output_dir
        ensure_dirs(
            [
                self.root_dir,
                self.root_dir / "checkpoints" / "regular",
                self.root_dir / "checkpoints" / "emergency",
                self.root_dir / "checkpoints" / "best",
                self.root_dir / "cache",
                self.root_dir / "logs" / "jsonl",
                self.root_dir / "logs" / "tensorboard",
            ]
        )
        if self.config.run.clear_cache:
            clear_cache_dirs(self.root_dir)

        self.logger = build_logger(self.config.project.name, self.config.project.verbosity)
        set_seed(self.config.project.seed)
        self.device = resolve_device(self.config.project.device)
        self.amp_dtype = configure_torch_runtime(config, self.device)
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.reward_shaper = RewardShaper(self.config.reward)
        self.telemetry = Telemetry(self.root_dir, self.config.logging)
        self.model = PolicyValueNet(self.config.model).to(self.device)
        self.model = maybe_compile(self.model, self.config)
        self.base_model = self._unwrap_model(self.model)
        self.optimizer = self._build_optimizer(stage="phase1")
        self.scaler = self._build_grad_scaler()
        self.teacher_buffer = RingBuffer(capacity=max(self.config.phase1.teacher_samples_per_epoch * 4, 50000))
        self.selfplay_buffer = RingBuffer(capacity=self.config.phase2.replay_buffer_capacity)
        self.adaptive_gae = self._build_adaptive_gae()
        self.current_stage = "phase1"
        self.epoch_in_stage = 0
        self.global_step = 0
        self.best_projected_elo = float("-inf")
        self.last_emergency_save_time = time.time()
        self.teacher: Optional[StockfishTeacher] = None
        self._recent_epoch_times: Dict[str, Deque[float]] = {
            "phase1": deque(maxlen=3),
            "phase2": deque(maxlen=3),
        }
        self._profiled_tflops: Dict[str, bool] = {"phase1": False, "phase2": False}
        self._phase_tflops_estimate: Dict[str, float] = {"phase1": 0.0, "phase2": 0.0}
        self._phase1_move_injection_probs = self._normalized_phase1_move_injection_probs()

        self._log_run_header()

    def _unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return getattr(model, "_orig_mod", model)

    def _build_optimizer(self, stage: str) -> torch.optim.Optimizer:
        stage_cfg = self.config.phase1 if stage == "phase1" else self.config.phase2
        return torch.optim.AdamW(
            self._unwrap_model(self.model).parameters(),
            lr=stage_cfg.lr,
            weight_decay=stage_cfg.weight_decay,
            betas=(0.9, 0.95),
            eps=1.0e-8,
        )

    def _build_grad_scaler(self) -> torch.amp.GradScaler:
        scaler_enabled = self.device.type == "cuda" and self.amp_dtype == torch.float16
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        return torch.cuda.amp.GradScaler(enabled=scaler_enabled)  # pragma: no cover

    def _build_adaptive_gae(self) -> AdaptiveGAE:
        return AdaptiveGAE(
            lambda_min=self.config.gae.lambda_min,
            lambda_max=self.config.gae.lambda_max,
            current_lambda=self.config.gae.lambda_init,
            ema_beta=self.config.gae.ema_beta,
            k_noise=self.config.gae.k_noise,
            k_value_var=self.config.gae.k_value_var,
        )

    def _maybe_open_teacher(self) -> Optional[StockfishTeacher]:
        if not self.config.teacher.enabled:
            return None
        if self.teacher is None:
            self.teacher = StockfishTeacher(self.config.teacher, self.move_encoder)
        return self.teacher

    def close(self) -> None:
        self.telemetry.close()
        if self.teacher is not None:
            self.teacher.close()
            self.teacher = None

    def save_checkpoint(self, emergency: bool = False) -> Path:
        checkpoint_dir = self.root_dir / "checkpoints" / ("emergency" if emergency else "regular")
        name = f"{self.current_stage}_epoch_{self.epoch_in_stage:04d}_step_{self.global_step:08d}.pt"
        path = checkpoint_dir / name
        payload = {
            "config": self.config.to_dict(),
            "current_stage": self.current_stage,
            "epoch_in_stage": self.epoch_in_stage,
            "global_step": self.global_step,
            "best_projected_elo": self.best_projected_elo,
            "model_state_dict": self.base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "teacher_buffer_state": self.teacher_buffer.state_dict(),
            "selfplay_buffer_state": self.selfplay_buffer.state_dict(),
            "adaptive_gae_state": self.adaptive_gae.state_dict(),
            "rng_state": get_rng_state(),
        }
        atomic_torch_save(payload, path)
        return path

    def safe_save_on_error(self, exc: Exception) -> None:
        if not self.config.run.safe_save_on_error:
            return
        path = self.save_checkpoint(emergency=True)
        self.logger.error("Emergency checkpoint saved to %s after exception: %s", path, exc)

    def maybe_periodic_emergency_save(self) -> None:
        every_minutes = max(self.config.logging.emergency_checkpoint_every_minutes, 1)
        if time.time() - self.last_emergency_save_time >= every_minutes * 60:
            self.save_checkpoint(emergency=True)
            self.last_emergency_save_time = time.time()

    def load_checkpoint(self, path: str | Path, checkpoint: Optional[Dict[str, object]] = None) -> None:
        if checkpoint is None:
            # PyTorch 2.6 defaults torch.load to weights_only=True.
            # Our trusted local checkpoints may include optimizer state and numpy objects.
            # Use weights_only=False for local resume/initialization payloads.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"{path} is not a valid checkpoint payload.")
        self.base_model.load_state_dict(checkpoint["model_state_dict"])
        self.current_stage = str(checkpoint["current_stage"])
        self.optimizer = self._build_optimizer(self.current_stage)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.teacher_buffer.load_state_dict(checkpoint["teacher_buffer_state"])
        self.selfplay_buffer.load_state_dict(checkpoint["selfplay_buffer_state"])
        self.adaptive_gae.load_state_dict(checkpoint["adaptive_gae_state"])
        self.epoch_in_stage = int(checkpoint["epoch_in_stage"])
        self.global_step = int(checkpoint["global_step"])
        self.best_projected_elo = float(checkpoint["best_projected_elo"])
        set_rng_state(checkpoint["rng_state"])
        self.logger.info("Resumed from %s | stage=%s epoch=%d", path, self.current_stage, self.epoch_in_stage)

    def load_model_weights_only(self, path: str | Path, checkpoint: Optional[Dict[str, object]] = None) -> None:
        if checkpoint is None:
            # PyTorch 2.6 defaults torch.load to weights_only=True.
            # Our trusted local checkpoints may include optimizer state and numpy objects.
            # Use weights_only=False for local resume/initialization payloads.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"{path} is not a valid checkpoint payload.")
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"{path} does not contain model_state_dict for weight-only initialization.")

        self.base_model.load_state_dict(checkpoint["model_state_dict"])
        target_stage = "phase2" if self.config.run.stage == "phase2" else "phase1"
        self.current_stage = target_stage
        self.epoch_in_stage = 0
        self.global_step = 0
        self.best_projected_elo = float(checkpoint.get("projected_elo", float("-inf")))
        self.optimizer = self._build_optimizer(target_stage)
        self.scaler = self._build_grad_scaler()
        self.teacher_buffer = RingBuffer(capacity=max(self.config.phase1.teacher_samples_per_epoch * 4, 50000))
        self.selfplay_buffer = RingBuffer(capacity=self.config.phase2.replay_buffer_capacity)
        self.adaptive_gae = self._build_adaptive_gae()
        self._recent_epoch_times["phase1"].clear()
        self._recent_epoch_times["phase2"].clear()
        self._profiled_tflops = {"phase1": False, "phase2": False}
        self._phase_tflops_estimate = {"phase1": 0.0, "phase2": 0.0}
        self.logger.info(
            "Loaded weight-only initialization from %s | target_stage=%s | optimizer/scaler/buffers reset",
            path,
            target_stage,
        )

    def phase2_teacher_ratio(self, phase2_epoch: int) -> float:
        start = self.config.curriculum.overlap_start_epoch
        span = max(self.config.curriculum.overlap_span_epochs, 1)
        if phase2_epoch < start:
            return self.config.curriculum.teacher_ratio_start
        progress = min(max((phase2_epoch - start) / span, 0.0), 1.0)
        ratio_start = self.config.curriculum.teacher_ratio_start
        ratio_end = self.config.curriculum.teacher_ratio_end
        return float(ratio_start + progress * (ratio_end - ratio_start))

    def run(self) -> None:
        try:
            if self.current_stage == "phase1" and self.config.run.stage in {"phase1", "full"}:
                self._run_phase1()
            if self.config.run.stage in {"phase2", "full"}:
                if self.current_stage != "phase2":
                    self.current_stage = "phase2"
                    self.epoch_in_stage = 0
                    self.optimizer = self._build_optimizer("phase2")
                self._run_phase2()
        except Exception as exc:
            self.safe_save_on_error(exc)
            raise
        finally:
            self.close()

    # -------------------------------------------------------------------------
    # Pretty run header and table helpers
    # -------------------------------------------------------------------------

    def _runtime_snapshot(self) -> Dict[str, str]:
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_mem_gb = gpu_props.total_memory / (1024 ** 3)
        else:
            gpu_name = "CPU"
            gpu_mem_gb = 0.0

        return {
            "device": str(self.device),
            "dtype": str(self.amp_dtype).replace("torch.", ""),
            "compile": str(self.config.project.compile),
            "compile_mode": str(self.config.project.compile_mode),
            "tf32": str(self.config.project.allow_tf32),
            "gpu_name": gpu_name,
            "gpu_mem_gb": f"{gpu_mem_gb:.1f}",
            "cpu_threads": str(torch.get_num_threads()),
            "platform": platform.platform(),
            "params_m": f"{count_parameters(self.base_model) / 1e6:.2f}",
        }

    def _estimate_initial_phase_time_seconds(self, phase: str) -> float:
        if phase == "phase1":
            train_steps = max(math.ceil(self.config.phase1.teacher_samples_per_epoch / self.config.phase1.batch_size), 1)
            train_seconds = train_steps * 0.30
            teacher_positions = self.config.phase1.teacher_games_per_epoch * self.config.phase1.max_teacher_game_plies
            teacher_seconds = (
                teacher_positions
                * (max(self.config.teacher.movetime_ms, 1) / 1000.0)
                * 1.15
            )
            epoch_seconds = max(train_seconds + teacher_seconds, 20.0)
            return epoch_seconds * self.config.phase1.epochs

        selfplay_positions = self.config.phase2.selfplay_games_per_epoch * self.config.phase2.max_game_plies * 0.60
        estimated_nodes = selfplay_positions * self.config.mcts.simulations
        selfplay_seconds = estimated_nodes / 2_500_000.0
        train_steps = max(self.config.phase2.minibatches_per_epoch * self.config.phase2.update_epochs_per_cycle, 1)
        train_seconds = train_steps * 0.45
        epoch_seconds = max(selfplay_seconds + train_seconds, 45.0)
        return epoch_seconds * self.config.phase2.epochs

    def _normalized_phase1_move_injection_probs(self) -> Tuple[float, float, float]:
        raw = np.array(
            [
                self.config.phase1.move_injection_best_prob,
                self.config.phase1.move_injection_topk_prob,
                self.config.phase1.move_injection_random_prob,
            ],
            dtype=np.float64,
        )
        clipped = np.clip(raw, 0.0, None)
        total = float(clipped.sum())
        if total <= 0.0:
            self.logger.warning("Invalid phase1 move injection probabilities; falling back to best-move only.")
            return 1.0, 0.0, 0.0

        normalized = clipped / total
        if not np.allclose(raw, normalized, atol=1e-6):
            self.logger.warning(
                "Normalized phase1 move injection probabilities to best=%.3f topk=%.3f random=%.3f",
                normalized[0],
                normalized[1],
                normalized[2],
            )
        return float(normalized[0]), float(normalized[1]), float(normalized[2])

    def _phase1_opening_noise_range(self) -> Tuple[int, int]:
        if not self.config.phase1.robust_distillation:
            return 0, 5
        lower = max(int(self.config.phase1.opening_noise_min_plies), 0)
        upper = max(int(self.config.phase1.opening_noise_max_plies), lower)
        return lower, upper

    def _record_epoch_time(self, phase: str, epoch_time_s: float) -> None:
        self._recent_epoch_times[phase].append(float(epoch_time_s))

    def _rolling_phase_eta_seconds(self, phase: str, remaining_epochs: int) -> Optional[float]:
        if remaining_epochs <= 0:
            return 0.0
        history = self._recent_epoch_times[phase]
        if len(history) < 2:
            return None
        avg_epoch_s = sum(history) / len(history)
        return avg_epoch_s * remaining_epochs

    def _eta_display(self, eta_seconds: Optional[float]) -> str:
        if eta_seconds is None:
            return "warming up"
        return format_seconds(max(eta_seconds, 0.0))

    def _log_run_header(self) -> None:
        rt = self._runtime_snapshot()
        phase1_eta = self._estimate_initial_phase_time_seconds("phase1")
        phase2_eta = self._estimate_initial_phase_time_seconds("phase2")
        opening_min, opening_max = self._phase1_opening_noise_range()
        p_best, p_topk, p_random = self._phase1_move_injection_probs

        self.logger.info("")
        self.logger.info("=" * 72)
        self.logger.info("Chess Hybrid RL Run Initialization")
        self.logger.info("=" * 72)
        self.logger.info(
            "Run: stage=%s | continue_from=%s | clear_cache=%s | profile=%s",
            self.config.run.stage,
            self.config.run.continue_from or "<none>",
            str(self.config.run.clear_cache),
            str(self.config.run.profile),
        )
        self.logger.info(
            "Device=%s | GPU=%s | VRAM=%sGB | dtype=%s | compile=%s (%s) | TF32=%s",
            rt["device"],
            rt["gpu_name"],
            rt["gpu_mem_gb"],
            rt["dtype"],
            rt["compile"],
            rt["compile_mode"],
            rt["tf32"],
        )
        self.logger.info(
            "CPU threads=%s | Platform=%s",
            rt["cpu_threads"],
            rt["platform"],
        )
        self.logger.info(
            "Model: params=%sM | blocks=%d | channels=%d | input_planes=%d | action_size=%d",
            rt["params_m"],
            self.config.model.num_blocks,
            self.config.model.channels,
            self.config.model.input_planes,
            self.config.model.action_size,
        )
        self.logger.info(
            "Phase config: phase1_epochs=%d | phase2_epochs=%d | curriculum_overlap=%d/%d",
            self.config.phase1.epochs,
            self.config.phase2.epochs,
            self.config.curriculum.overlap_start_epoch,
            self.config.curriculum.overlap_span_epochs,
        )
        self.logger.info(
            "Phase I: batch=%d | lr=%g | teacher_samples=%d | teacher_games=%d | max_plies=%d",
            self.config.phase1.batch_size,
            self.config.phase1.lr,
            self.config.phase1.teacher_samples_per_epoch,
            self.config.phase1.teacher_games_per_epoch,
            self.config.phase1.max_teacher_game_plies,
        )
        self.logger.info(
            "Phase I.5 robust distill: enabled=%s | opening_noise=%d-%d plies | mix(best/topk/random)=%.2f/%.2f/%.2f | topk=%d",
            str(self.config.phase1.robust_distillation),
            opening_min,
            opening_max,
            p_best,
            p_topk,
            p_random,
            self.config.phase1.move_injection_topk,
        )
        self.logger.info(
            "Phase II: batch=%d | lr=%g | selfplay_games=%d | replay=%d | minibatches=%d | update_epochs=%d",
            self.config.phase2.batch_size,
            self.config.phase2.lr,
            self.config.phase2.selfplay_games_per_epoch,
            self.config.phase2.replay_buffer_capacity,
            self.config.phase2.minibatches_per_epoch,
            self.config.phase2.update_epochs_per_cycle,
        )
        self.logger.info(
            "MCTS: sims=%d | c_puct=%.2f | temperature=%.2f->%.2f | root_noise=%s",
            self.config.mcts.simulations,
            self.config.mcts.c_puct,
            self.config.mcts.temperature_start,
            self.config.mcts.temperature_end,
            str(self.config.mcts.root_noise),
        )
        self.logger.info(
            "Teacher: enabled=%s | engine=%s | threads=%d | hash=%dMB | movetime=%dms | multipv=%d",
            str(self.config.teacher.enabled),
            self.config.teacher.engine_path,
            self.config.teacher.threads,
            self.config.teacher.hash_mb,
            self.config.teacher.movetime_ms,
            self.config.teacher.multipv,
        )
        self.logger.info(
            "GAE: gamma=%.4f | lambda_init=%.3f | lambda_range=[%.2f, %.2f] | ema_beta=%.2f",
            self.config.gae.gamma,
            self.config.gae.lambda_init,
            self.config.gae.lambda_min,
            self.config.gae.lambda_max,
            self.config.gae.ema_beta,
        )
        self.logger.info("TFLOPs metric: sparse profiler estimate (approximate, one sample per phase)")
        self.logger.info(
            "Initial ETA Heuristic: Phase I=%s | Phase II=%s | Total=%s",
            format_seconds(phase1_eta),
            format_seconds(phase2_eta),
            format_seconds(phase1_eta + phase2_eta),
        )
        self.logger.info("=" * 72)

    def _table_separator(self) -> str:
        columns = (10, 8, 8, 8, 8, 10, 10, 8, 10)
        return "+" + "+".join("-" * (width + 2) for width in columns) + "+"

    def _phase_table_header(self, phase: str) -> None:
        self.logger.info("")
        self.logger.info("Phase %s live epoch table", phase)
        self.logger.info(self._table_separator())
        self.logger.info(
            "| {epoch:^10} | {loss:^8} | {pol:^8} | {val:^8} | {tflops:^8} | {pos:^10} | {nodes:^10} | {time:^8} | {eta:^10} |".format(
                epoch="Epoch",
                loss="Loss",
                pol="PolLoss",
                val="ValLoss",
                tflops="TFLOPs",
                pos="Pos/s",
                nodes="Nodes/s",
                time="Time",
                eta="ETA",
            )
        )
        self.logger.info(self._table_separator())

    def _phase_table_row(
        self,
        epoch_display: str,
        loss_total: float,
        loss_policy: float,
        loss_value: float,
        tflops: float,
        pos_per_s: float,
        nodes_per_s: float,
        epoch_time_s: float,
        eta_display: str,
    ) -> None:
        self.logger.info(
            "| {epoch:<10} | {loss:>8.4f} | {pol:>8.4f} | {val:>8.4f} | {tflops:>8.2f} | {pos:>10.1f} | {nodes:>10.1f} | {etime:>8} | {eta:>10} |".format(
                epoch=epoch_display,
                loss=loss_total,
                pol=loss_policy,
                val=loss_value,
                tflops=tflops,
                pos=pos_per_s,
                nodes=nodes_per_s,
                etime=format_seconds(epoch_time_s),
                eta=eta_display,
            )
        )

    def _maybe_profile_tflops(
        self,
        stage: str,
        batch_idx: int,
        step_fn: Callable[[], torch.Tensor],
        update_epoch: int = 0,
    ) -> None:
        if not self.config.run.profile:
            return
        if self._profiled_tflops.get(stage, False):
            return
        if batch_idx != 0 or update_epoch != 0:
            return

        measured_tflops = 0.0
        self.logger.info("Sampling profiler-based TFLOPs for %s (one-time, approximate).", stage)
        try:
            self.optimizer.zero_grad(set_to_none=True)
            measured_tflops = float(estimate_tflops_from_profile(step_fn, self.device))
            if not math.isfinite(measured_tflops) or measured_tflops < 0.0:
                measured_tflops = 0.0
        except Exception as exc:
            self.logger.warning("TFLOPs profiling failed for %s: %s. Using 0.0.", stage, exc)
            measured_tflops = 0.0
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self._profiled_tflops[stage] = True
            self._phase_tflops_estimate[stage] = measured_tflops

    # -------------------------------------------------------------------------
    # Phase loops
    # -------------------------------------------------------------------------

    def _run_phase1(self) -> None:
        target_phase1_epochs = self.config.phase1.epochs
        if self.config.run.continue_from and self.epoch_in_stage >= self.config.phase1.epochs:
            target_phase1_epochs = self.epoch_in_stage + self.config.phase1.epochs
            self.logger.info(
                "Phase I continue mode: start_epoch=%d, configured_epochs=%d -> target_epoch=%d",
                self.epoch_in_stage,
                self.config.phase1.epochs,
                target_phase1_epochs,
            )

        self.logger.info(
            "Starting Phase I distillation | start_epoch=%d | target_epoch=%d",
            self.epoch_in_stage,
            target_phase1_epochs,
        )
        if not self.config.run.profile:
            self.logger.info("TFLOPs profiling disabled (run.profile=false); table will report 0.00 estimates.")
        self._phase_table_header("I")
        self._recent_epoch_times["phase1"].clear()

        for epoch in range(self.epoch_in_stage, target_phase1_epochs):
            self.current_stage = "phase1"
            self.epoch_in_stage = epoch
            epoch_start = time.perf_counter()

            self._refresh_teacher_buffer()
            metrics = self._train_distillation_epoch(epoch)

            epoch_time_s = time.perf_counter() - epoch_start
            self._record_epoch_time("phase1", epoch_time_s)
            remaining_epochs = target_phase1_epochs - (epoch + 1)
            eta_s = self._rolling_phase_eta_seconds("phase1", remaining_epochs)
            eta_display = self._eta_display(eta_s)

            self.global_step += 1
            self.telemetry.log_metrics(self.global_step, {f"phase1/{k}": v for k, v in metrics.items()})

            self._phase_table_row(
                epoch_display=f"{epoch + 1}/{target_phase1_epochs}",
                loss_total=metrics["loss_total"],
                loss_policy=metrics["loss_policy"],
                loss_value=metrics["loss_value"],
                tflops=metrics.get("tflops", 0.0),
                pos_per_s=metrics.get("positions_per_sec", 0.0),
                nodes_per_s=metrics.get("nodes_per_sec", 0.0),
                epoch_time_s=epoch_time_s,
                eta_display=eta_display,
            )

            self.logger.debug(
                "Phase I epoch %d summary | loss=%.4f | pol=%.4f | val=%.4f | entropy=%.4f | grad=%.4f | tflops=%.2f | pos/s=%.2f | epoch_time=%s | phase_eta=%s",
                epoch + 1,
                metrics["loss_total"],
                metrics["loss_policy"],
                metrics["loss_value"],
                metrics["policy_entropy"],
                metrics["grad_norm"],
                metrics.get("tflops", 0.0),
                metrics.get("positions_per_sec", 0.0),
                format_seconds(epoch_time_s),
                eta_display,
            )

            self.epoch_in_stage = epoch + 1
            if (epoch + 1) % self.config.logging.checkpoint_every_epochs == 0:
                self.save_checkpoint(emergency=False)
            self.maybe_periodic_emergency_save()

        self.logger.info(self._table_separator())
        self.current_stage = "phase2"
        self.epoch_in_stage = 0

    def _run_phase2(self) -> None:
        self.logger.info("Starting Phase II self-play for %d epochs", self.config.phase2.epochs)
        if not self.config.run.profile:
            self.logger.info("TFLOPs profiling disabled (run.profile=false); table will report 0.00 estimates.")
        self._phase_table_header("II")
        self._recent_epoch_times["phase2"].clear()

        for epoch in range(self.epoch_in_stage, self.config.phase2.epochs):
            self.current_stage = "phase2"
            self.epoch_in_stage = epoch
            epoch_start = time.perf_counter()

            teacher_ratio = self.phase2_teacher_ratio(epoch)
            if teacher_ratio > 0.0:
                self._refresh_teacher_buffer(limit=max(self.config.phase1.teacher_samples_per_epoch // 2, 2048))

            collection_metrics = self._collect_selfplay_epoch(teacher_ratio=teacher_ratio)
            training_metrics = self._train_selfplay_epoch(epoch, teacher_ratio=teacher_ratio)

            metrics = {
                **collection_metrics,
                **training_metrics,
                "teacher_ratio": teacher_ratio,
                "adaptive_lambda": self.adaptive_gae.current_lambda,
            }

            if teacher_ratio > 0.0:
                distill_metrics = self._train_distillation_epoch(
                    epoch,
                    max_batches=max(2, self.config.phase2.minibatches_per_epoch // 8),
                    log_prefix="phase2_distill",
                )
                metrics.update({f"distill_{k}": v for k, v in distill_metrics.items()})

            epoch_time_s = time.perf_counter() - epoch_start
            self._record_epoch_time("phase2", epoch_time_s)
            remaining_epochs = self.config.phase2.epochs - (epoch + 1)
            eta_s = self._rolling_phase_eta_seconds("phase2", remaining_epochs)
            eta_display = self._eta_display(eta_s)

            self.global_step += 1
            self.telemetry.log_metrics(self.global_step, {f"phase2/{k}": v for k, v in metrics.items()})

            self._phase_table_row(
                epoch_display=f"{epoch + 1}/{self.config.phase2.epochs}",
                loss_total=metrics["loss_total"],
                loss_policy=metrics["loss_policy"],
                loss_value=metrics["loss_value"],
                tflops=metrics.get("tflops", 0.0),
                pos_per_s=metrics.get("positions_per_sec", 0.0),
                nodes_per_s=metrics.get("nodes_per_sec", 0.0),
                epoch_time_s=epoch_time_s,
                eta_display=eta_display,
            )

            self.logger.debug(
                "Phase II epoch %d summary | loss=%.4f | pol=%.4f | val=%.4f | search=%.4f | entropy=%.4f | grad=%.4f | tflops=%.2f | lambda=%.4f | pos/s=%.2f | nodes/s=%.2f | teacher_ratio=%.3f | epoch_time=%s | phase_eta=%s",
                epoch + 1,
                metrics["loss_total"],
                metrics["loss_policy"],
                metrics["loss_value"],
                metrics["loss_search"],
                metrics["policy_entropy"],
                metrics["grad_norm"],
                metrics.get("tflops", 0.0),
                metrics["adaptive_lambda"],
                metrics.get("positions_per_sec", 0.0),
                metrics.get("nodes_per_sec", 0.0),
                teacher_ratio,
                format_seconds(epoch_time_s),
                eta_display,
            )

            self.epoch_in_stage = epoch + 1

            if (epoch + 1) % self.config.eval.every_n_epochs == 0:
                elo = self._evaluate_projected_elo()
                self.telemetry.log_metrics(self.global_step, {"phase2/projected_elo": elo})
                if elo > self.best_projected_elo:
                    self.best_projected_elo = elo
                    best_path = self.root_dir / "checkpoints" / "best" / "best_model.pt"
                    atomic_torch_save({"model_state_dict": self.base_model.state_dict(), "projected_elo": elo}, best_path)

            if (epoch + 1) % self.config.logging.checkpoint_every_epochs == 0:
                self.save_checkpoint(emergency=False)
            self.maybe_periodic_emergency_save()

        self.logger.info(self._table_separator())

    # -------------------------------------------------------------------------
    # Data refresh and loaders
    # -------------------------------------------------------------------------

    def _fallback_teacher_move(self, board: chess.Board, preferred_move: chess.Move) -> chess.Move:
        if preferred_move in board.legal_moves:
            return preferred_move
        legal_moves = list(board.legal_moves)
        return legal_moves[0]

    def _sample_teacher_play_move(self, board: chess.Board, analysis: TeacherAnalysis) -> chess.Move:
        fallback_move = self._fallback_teacher_move(board, analysis.best_move)
        if not self.config.phase1.robust_distillation:
            if analysis.policy.indices.size > 1:
                try:
                    move_index = int(np.random.choice(analysis.policy.indices, p=analysis.policy.probs))
                    candidate = self.move_encoder.index_to_move(move_index)
                    if candidate in board.legal_moves:
                        return candidate
                except Exception:
                    return fallback_move
            return fallback_move

        p_best, p_topk, p_random = self._phase1_move_injection_probs
        choice = int(np.random.choice(3, p=np.array([p_best, p_topk, p_random], dtype=np.float64)))
        if choice == 0:
            return fallback_move
        if choice == 2:
            return np.random.choice(list(board.legal_moves))

        topk = max(int(self.config.phase1.move_injection_topk), 1)
        candidate_indices = analysis.policy.indices[:topk]
        if candidate_indices.size == 0:
            return fallback_move

        candidate_probs = np.clip(analysis.policy.probs[: candidate_indices.size].astype(np.float64), 0.0, None)
        prob_sum = float(candidate_probs.sum())
        if prob_sum <= 0.0:
            candidate_probs = np.full(candidate_indices.size, 1.0 / candidate_indices.size, dtype=np.float64)
        else:
            candidate_probs = candidate_probs / prob_sum
        try:
            move_index = int(np.random.choice(candidate_indices, p=candidate_probs))
            candidate = self.move_encoder.index_to_move(move_index)
            if candidate in board.legal_moves:
                return candidate
        except Exception:
            return fallback_move
        return fallback_move

    def _refresh_teacher_buffer(self, limit: Optional[int] = None) -> None:
        teacher = self._maybe_open_teacher()
        if teacher is None:
            return
        target_samples = int(limit or self.config.phase1.teacher_samples_per_epoch)
        samples: List[TeacherSample] = []
        opening_min, opening_max = self._phase1_opening_noise_range()

        for _ in range(self.config.phase1.teacher_games_per_epoch):
            board = chess.Board()
            random_opening_plies = int(np.random.randint(opening_min, opening_max + 1))
            for _ in range(random_opening_plies):
                if board.is_game_over(claim_draw=True):
                    break
                move = np.random.choice(list(board.legal_moves))
                board.push(move)

            while not board.is_game_over(claim_draw=True) and len(samples) < target_samples and board.ply() < self.config.phase1.max_teacher_game_plies:
                analysis = teacher.analyze(board)
                sample = TeacherSample(
                    state=self.board_encoder.encode(board),
                    policy_indices=analysis.policy.indices.copy(),
                    policy_probs=analysis.policy.probs.copy(),
                    value_target=analysis.value,
                )
                samples.append(sample)

                move = self._sample_teacher_play_move(board, analysis)
                board.push(move)

            if len(samples) >= target_samples:
                break

        self.teacher_buffer.extend([item.to_dict() for item in samples])

    def _loader_kwargs(self) -> Dict:
        workers = max(self.config.hardware.num_workers, 0)
        kwargs = {
            "num_workers": workers,
            "pin_memory": self.config.hardware.pin_memory and self.device.type == "cuda",
            "persistent_workers": self.config.hardware.persistent_workers and workers > 0,
        }
        if workers > 0:
            kwargs["prefetch_factor"] = self.config.hardware.prefetch_factor
        return kwargs

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def _train_distillation_epoch(self, epoch: int, max_batches: Optional[int] = None, log_prefix: str = "phase1") -> Dict[str, float]:
        if len(self.teacher_buffer) == 0:
            raise RuntimeError("Teacher buffer is empty. Generate teacher data before training.")

        items = [TeacherSample.from_dict(item) for item in self.teacher_buffer.sample(self.config.phase1.teacher_samples_per_epoch)]
        dataset = TeacherDataset(items, self.move_encoder.action_size)
        loader = DataLoader(
            dataset,
            batch_size=self.config.phase1.batch_size,
            shuffle=True,
            drop_last=False,
            **self._loader_kwargs(),
        )

        self.model.train()
        meter = ThroughputMeter()
        sums = {
            "loss_total": 0.0,
            "loss_policy": 0.0,
            "loss_value": 0.0,
            "policy_entropy": 0.0,
            "grad_norm": 0.0,
        }
        batch_count = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = to_device(batch, self.device)
            states = batch["states"]
            policy_targets = batch["policy_targets"]
            value_targets = batch["value_targets"]
            meter.update(positions=int(states.size(0)), nodes=0)

            def forward_only() -> torch.Tensor:
                output = self.model(states)
                policy_log_probs = F.log_softmax(output.policy_logits, dim=-1)
                policy_loss = -(policy_targets * policy_log_probs).sum(dim=-1).mean()
                value_loss = F.mse_loss(output.value, value_targets)
                entropy = policy_entropy_from_logits(output.policy_logits)
                return (
                    self.config.phase1.policy_loss_weight * policy_loss
                    + self.config.phase1.value_loss_weight * value_loss
                    - self.config.phase1.entropy_weight * entropy
                )

            self.optimizer.zero_grad(set_to_none=True)
            autocast_enabled = self.device.type == "cuda" and self.amp_dtype in (torch.float16, torch.bfloat16)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=autocast_enabled):
                output = self.model(states)
                policy_log_probs = F.log_softmax(output.policy_logits, dim=-1)
                policy_loss = -(policy_targets * policy_log_probs).sum(dim=-1).mean()
                value_loss = F.mse_loss(output.value, value_targets)
                entropy = policy_entropy_from_logits(output.policy_logits)
                loss = (
                    self.config.phase1.policy_loss_weight * policy_loss
                    + self.config.phase1.value_loss_weight * value_loss
                    - self.config.phase1.entropy_weight * entropy
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # TFLOPs is a sparse profiler-based estimate, not a per-batch ground-truth metric.
            self._maybe_profile_tflops(stage=self.current_stage, batch_idx=batch_idx, step_fn=forward_only)

            sums["loss_total"] += float(loss.item())
            sums["loss_policy"] += float(policy_loss.item())
            sums["loss_value"] += float(value_loss.item())
            sums["policy_entropy"] += float(entropy.item())
            sums["grad_norm"] += float(grad_norm)
            batch_count += 1

        steps = max(batch_count, 1)
        metrics = {key: value / steps for key, value in sums.items()}
        metrics["tflops"] = float(self._phase_tflops_estimate.get(self.current_stage, 0.0))
        metrics.update(meter.summary())
        metrics.update(device_metrics(self.device))
        return metrics

    def _policy_and_value(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        self.model.eval()
        state = torch.from_numpy(self.board_encoder.encode(board)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            autocast_enabled = self.device.type == "cuda" and self.amp_dtype in (torch.float16, torch.bfloat16)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=autocast_enabled):
                output = self.model(state)
            logits = output.policy_logits[0]
            value = float(output.value[0].cpu().item())

        legal_mask = self.move_encoder.legal_mask(board).to(logits.device)
        probs = torch.softmax(logits.masked_fill(~legal_mask, float("-inf")), dim=-1)
        return probs.cpu().to(torch.float32).numpy(), value

    def _collect_selfplay_epoch(self, teacher_ratio: float) -> Dict[str, float]:
        teacher = self._maybe_open_teacher() if teacher_ratio > 0.0 else None
        mcts = MCTS(self.model, self.board_encoder, self.move_encoder, self.config.mcts, self.device, self.amp_dtype)
        meter = ThroughputMeter()
        all_transitions: List[SelfPlayTransition] = []
        wins = draws = losses = 0

        for _ in range(self.config.phase2.selfplay_games_per_epoch):
            board = chess.Board()
            episode: List[SelfPlayTransition] = []
            random_opening_plies = np.random.randint(0, 6)

            for _ in range(random_opening_plies):
                if board.is_game_over(claim_draw=True):
                    break
                board.push(np.random.choice(list(board.legal_moves)))

            while not board.is_game_over(claim_draw=True) and board.ply() < self.config.phase2.max_game_plies:
                actor_color = board.turn
                state = self.board_encoder.encode(board)
                raw_policy, value_pred = self._policy_and_value(board)
                search = mcts.run(board, ply=board.ply())
                meter.update(positions=1, nodes=search.nodes)

                action = search.action
                move = self.move_encoder.index_to_move(action)
                if move not in board.legal_moves:
                    move = next(iter(board.legal_moves))
                    action = self.move_encoder.move_to_index(move)

                dense_reward = 0.0
                if teacher is not None:
                    before = teacher.analyze(board)
                    action_matches = action == self.move_encoder.move_to_index(before.best_move)
                    next_board = board.copy(stack=False)
                    next_board.push(move)
                    after_cp = before.cp
                    if not next_board.is_game_over(claim_draw=True):
                        after = teacher.analyze(next_board)
                        after_cp = after.cp
                    dense_reward = self.reward_shaper.dense_reward(
                        before.cp,
                        after_cp,
                        action_matches,
                        3 if board.is_repetition(3) else 1,
                    )

                behavior_log_prob = float(np.log(max(raw_policy[action], 1e-9)))
                board.push(move)
                terminal = self.reward_shaper.terminal_reward(board, actor_color, board.ply())
                reward = teacher_ratio * dense_reward + terminal.reward
                next_state = None if terminal.terminal else self.board_encoder.encode(board)

                episode.append(
                    SelfPlayTransition(
                        state=state,
                        action=action,
                        search_policy_indices=search.policy.indices.copy(),
                        search_policy_probs=search.policy.probs.copy(),
                        reward=reward,
                        value_pred=value_pred,
                        log_prob=behavior_log_prob,
                        done=terminal.terminal,
                        next_state=next_state,
                        player_sign=1.0 if actor_color == chess.WHITE else -1.0,
                    )
                )

                if terminal.terminal:
                    if terminal.is_draw:
                        draws += 1
                    elif terminal.result > 0:
                        wins += 1
                    else:
                        losses += 1
                    break

            if episode and not episode[-1].done:
                episode[-1].done = True
                episode[-1].reward += self.reward_shaper.draw_reward(board.ply())
                draws += 1

            self._compute_advantages(episode)
            all_transitions.extend(episode)

        self.selfplay_buffer.extend([item.to_dict() for item in all_transitions])

        summary = meter.summary()
        total_games = max(self.config.phase2.selfplay_games_per_epoch, 1)
        summary.update(
            {
                "wins": wins / total_games,
                "draws": draws / total_games,
                "losses": losses / total_games,
                "games_collected": total_games,
                "transitions_collected": len(all_transitions),
            }
        )
        return summary

    def _compute_advantages(self, episode: Sequence[SelfPlayTransition]) -> None:
        if not episode:
            return

        gamma = self.config.gae.gamma
        values = np.array([t.value_pred for t in episode], dtype=np.float32)
        next_values = np.zeros_like(values)

        for idx, transition in enumerate(episode[:-1]):
            next_values[idx] = -episode[idx + 1].value_pred
            if transition.done:
                next_values[idx] = 0.0
        next_values[-1] = 0.0

        rewards = np.array([t.reward for t in episode], dtype=np.float32)
        dones = np.array([t.done for t in episode], dtype=np.float32)
        td_errors = rewards + gamma * (1.0 - dones) * next_values - values

        lam = self.adaptive_gae.update(values, td_errors)
        advantages = np.zeros_like(values)
        gae = 0.0

        for idx in reversed(range(len(episode))):
            gae = td_errors[idx] + gamma * lam * (1.0 - dones[idx]) * gae
            advantages[idx] = gae

        returns = advantages + values

        for idx, transition in enumerate(episode):
            transition.advantage = float(advantages[idx])
            transition.return_target = float(returns[idx])

    def _train_selfplay_epoch(self, epoch: int, teacher_ratio: float) -> Dict[str, float]:
        if len(self.selfplay_buffer) == 0:
            raise RuntimeError("Self-play buffer is empty. Collect self-play before training.")

        sample_count = min(
            len(self.selfplay_buffer),
            self.config.phase2.batch_size * self.config.phase2.minibatches_per_epoch,
        )
        items = [SelfPlayTransition.from_dict(item) for item in self.selfplay_buffer.sample(sample_count)]

        advantages = np.array([item.advantage for item in items], dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for idx, item in enumerate(items):
            item.advantage = float(advantages[idx])

        dataset = SelfPlayDataset(items, self.move_encoder.action_size)
        loader = DataLoader(
            dataset,
            batch_size=self.config.phase2.batch_size,
            shuffle=True,
            drop_last=False,
            **self._loader_kwargs(),
        )

        self.model.train()
        sums = {
            "loss_total": 0.0,
            "loss_policy": 0.0,
            "loss_search": 0.0,
            "loss_value": 0.0,
            "policy_entropy": 0.0,
            "grad_norm": 0.0,
        }
        batch_count = 0

        for update_epoch in range(self.config.phase2.update_epochs_per_cycle):
            for batch_idx, batch in enumerate(loader):
                batch = to_device(batch, self.device)
                states = batch["states"]
                actions = batch["actions"]
                returns = batch["returns"]
                search_policy = batch["search_policy"]
                advantages_batch = batch["advantages"]

                def forward_only() -> torch.Tensor:
                    output = self.model(states)
                    log_probs = F.log_softmax(output.policy_logits, dim=-1)
                    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                    policy_loss = -(advantages_batch.detach() * action_log_probs).mean()
                    search_loss = -(search_policy * log_probs).sum(dim=-1).mean()
                    value_loss = F.mse_loss(output.value, returns)
                    entropy = policy_entropy_from_logits(output.policy_logits)
                    return (
                        self.config.phase2.policy_loss_weight * policy_loss
                        + self.config.phase2.search_loss_weight * search_loss
                        + self.config.phase2.value_loss_weight * value_loss
                        - self.config.phase2.entropy_weight * entropy
                    )

                self.optimizer.zero_grad(set_to_none=True)
                autocast_enabled = self.device.type == "cuda" and self.amp_dtype in (torch.float16, torch.bfloat16)

                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=autocast_enabled):
                    output = self.model(states)
                    log_probs = F.log_softmax(output.policy_logits, dim=-1)
                    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                    policy_loss = -(advantages_batch.detach() * action_log_probs).mean()
                    search_loss = -(search_policy * log_probs).sum(dim=-1).mean()
                    value_loss = F.mse_loss(output.value, returns)
                    entropy = policy_entropy_from_logits(output.policy_logits)
                    loss = (
                        self.config.phase2.policy_loss_weight * policy_loss
                        + self.config.phase2.search_loss_weight * search_loss
                        + self.config.phase2.value_loss_weight * value_loss
                        - self.config.phase2.entropy_weight * entropy
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # TFLOPs is sampled once per phase to keep profiler overhead low.
                self._maybe_profile_tflops(
                    stage=self.current_stage,
                    batch_idx=batch_idx,
                    update_epoch=update_epoch,
                    step_fn=forward_only,
                )

                sums["loss_total"] += float(loss.item())
                sums["loss_policy"] += float(policy_loss.item())
                sums["loss_search"] += float(search_loss.item())
                sums["loss_value"] += float(value_loss.item())
                sums["policy_entropy"] += float(entropy.item())
                sums["grad_norm"] += float(grad_norm)
                batch_count += 1

        steps = max(batch_count, 1)
        metrics = {key: value / steps for key, value in sums.items()}
        metrics["tflops"] = float(self._phase_tflops_estimate.get(self.current_stage, 0.0))
        metrics.update(device_metrics(self.device))
        return metrics

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def _agent_move(self, board: chess.Board, mcts: MCTS) -> chess.Move:
        search = mcts.run(board, board.ply())
        move = self.move_encoder.index_to_move(search.action)
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        return move

    def _evaluate_projected_elo(self) -> float:
        teacher_cfg = copy.deepcopy(self.config.teacher)
        teacher_cfg.movetime_ms = self.config.eval.baseline_movetime_ms
        score = 0.0
        mcts = MCTS(self.model, self.board_encoder, self.move_encoder, self.config.mcts, self.device, self.amp_dtype)

        with StockfishTeacher(teacher_cfg, self.move_encoder) as opponent:
            for game_idx in range(self.config.eval.arena_games):
                board = chess.Board()
                agent_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK

                while not board.is_game_over(claim_draw=True) and board.ply() < self.config.eval.max_game_plies:
                    if board.turn == agent_color:
                        move = self._agent_move(board, mcts)
                    else:
                        move = opponent.analyze(board).best_move
                    board.push(move)

                outcome = board.outcome(claim_draw=True)
                if outcome is None or outcome.winner is None:
                    score += 0.5
                elif outcome.winner == agent_color:
                    score += 1.0

        score_fraction = score / max(self.config.eval.arena_games, 1)
        elo = projected_elo(score_fraction, self.config.eval.opponent_elo)
        self.logger.info("Projected Elo %.1f from score fraction %.3f", elo, score_fraction)
        return elo
