from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from .teacher import StockfishTeacher
from .telemetry import Telemetry, ThroughputMeter, build_logger, device_metrics, estimate_tflops_from_profile, policy_entropy_from_logits, projected_elo
from .utils import (
    atomic_torch_save,
    clear_cache_dirs,
    configure_torch_runtime,
    count_parameters,
    ensure_dirs,
    format_seconds,
    get_rng_state,
    grad_global_norm,
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
        scaler_enabled = self.device.type == "cuda" and self.amp_dtype == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.teacher_buffer = RingBuffer(capacity=max(self.config.phase1.teacher_samples_per_epoch * 4, 50000))
        self.selfplay_buffer = RingBuffer(capacity=self.config.phase2.replay_buffer_capacity)
        self.adaptive_gae = AdaptiveGAE(
            lambda_min=self.config.gae.lambda_min,
            lambda_max=self.config.gae.lambda_max,
            current_lambda=self.config.gae.lambda_init,
            ema_beta=self.config.gae.ema_beta,
            k_noise=self.config.gae.k_noise,
            k_value_var=self.config.gae.k_value_var,
        )
        self.current_stage = "phase1"
        self.epoch_in_stage = 0
        self.global_step = 0
        self.best_projected_elo = float("-inf")
        self.last_emergency_save_time = time.time()
        self.teacher: Optional[StockfishTeacher] = None
        self.logger.info(
            "Initialized on %s | dtype=%s | params=%d",
            self.device,
            str(self.amp_dtype),
            count_parameters(self.base_model),
        )

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

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
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
        except Exception as exc:  # pragma: no cover
            self.safe_save_on_error(exc)
            raise
        finally:
            self.close()

    def _run_phase1(self) -> None:
        self.logger.info("Starting Phase I distillation for %d epochs", self.config.phase1.epochs)
        for epoch in range(self.epoch_in_stage, self.config.phase1.epochs):
            self.current_stage = "phase1"
            self.epoch_in_stage = epoch
            self._refresh_teacher_buffer()
            metrics = self._train_distillation_epoch(epoch)
            self.global_step += 1
            self.telemetry.log_metrics(self.global_step, {f"phase1/{k}": v for k, v in metrics.items()})
            self.logger.info("Phase I epoch %d | loss=%.4f | positions/s=%.2f", epoch, metrics["loss_total"], metrics["positions_per_sec"])
            self.epoch_in_stage = epoch + 1
            if (epoch + 1) % self.config.logging.checkpoint_every_epochs == 0:
                self.save_checkpoint(emergency=False)
            self.maybe_periodic_emergency_save()
        self.current_stage = "phase2"
        self.epoch_in_stage = 0

    def _run_phase2(self) -> None:
        self.logger.info("Starting Phase II self-play for %d epochs", self.config.phase2.epochs)
        for epoch in range(self.epoch_in_stage, self.config.phase2.epochs):
            self.current_stage = "phase2"
            self.epoch_in_stage = epoch
            teacher_ratio = self.phase2_teacher_ratio(epoch)
            if teacher_ratio > 0.0:
                self._refresh_teacher_buffer(limit=max(self.config.phase1.teacher_samples_per_epoch // 2, 2048))
            collection_metrics = self._collect_selfplay_epoch(teacher_ratio=teacher_ratio)
            training_metrics = self._train_selfplay_epoch(epoch, teacher_ratio=teacher_ratio)
            metrics = {**collection_metrics, **training_metrics, "teacher_ratio": teacher_ratio, "adaptive_lambda": self.adaptive_gae.current_lambda}
            if teacher_ratio > 0.0:
                distill_metrics = self._train_distillation_epoch(epoch, max_batches=max(2, self.config.phase2.minibatches_per_epoch // 8), log_prefix="phase2_distill")
                metrics.update({f"distill_{k}": v for k, v in distill_metrics.items()})
            self.global_step += 1
            self.telemetry.log_metrics(self.global_step, {f"phase2/{k}": v for k, v in metrics.items()})
            self.epoch_in_stage = epoch + 1
            self.logger.info(
                "Phase II epoch %d | loss=%.4f | lambda=%.4f | pos/s=%.2f | nodes/s=%.2f",
                epoch,
                metrics["loss_total"],
                metrics["adaptive_lambda"],
                metrics["positions_per_sec"],
                metrics["nodes_per_sec"],
            )
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

    def _refresh_teacher_buffer(self, limit: Optional[int] = None) -> None:
        teacher = self._maybe_open_teacher()
        if teacher is None:
            return
        target_samples = int(limit or self.config.phase1.teacher_samples_per_epoch)
        samples: List[TeacherSample] = []
        for _ in range(self.config.phase1.teacher_games_per_epoch):
            board = chess.Board()
            random_opening_plies = np.random.randint(0, 6)
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
                if analysis.policy.indices.size > 1:
                    move_index = int(np.random.choice(analysis.policy.indices, p=analysis.policy.probs))
                    move = self.move_encoder.index_to_move(move_index)
                    if move not in board.legal_moves:
                        move = analysis.best_move
                else:
                    move = analysis.best_move
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

    def _train_distillation_epoch(self, epoch: int, max_batches: Optional[int] = None, log_prefix: str = "phase1") -> Dict[str, float]:
        if len(self.teacher_buffer) == 0:
            raise RuntimeError("Teacher buffer is empty. Generate teacher data before training.")
        items = [TeacherSample.from_dict(item) for item in self.teacher_buffer.sample(self.config.phase1.teacher_samples_per_epoch)]
        dataset = TeacherDataset(items, self.move_encoder.action_size)
        loader = DataLoader(dataset, batch_size=self.config.phase1.batch_size, shuffle=True, drop_last=False, **self._loader_kwargs())
        self.model.train()
        meter = ThroughputMeter()
        sums = {"loss_total": 0.0, "loss_policy": 0.0, "loss_value": 0.0, "policy_entropy": 0.0, "grad_norm": 0.0, "tflops": 0.0}
        profiled = False
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

            tflops = 0.0
            if self.config.run.profile and not profiled and (batch_idx % max(self.config.logging.profile_every_steps, 1) == 0):
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                    tflops = estimate_tflops_from_profile(lambda: forward_only(), self.device)
                    profiled = True
                finally:
                    self.optimizer.zero_grad(set_to_none=True)

            sums["loss_total"] += float(loss.item())
            sums["loss_policy"] += float(policy_loss.item())
            sums["loss_value"] += float(value_loss.item())
            sums["policy_entropy"] += float(entropy.item())
            sums["grad_norm"] += float(grad_norm)
            sums["tflops"] += tflops

        steps = max(batch_idx + 1, 1)
        metrics = {key: value / steps for key, value in sums.items()}
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
                    dense_reward = self.reward_shaper.dense_reward(before.cp, after_cp, action_matches, 3 if board.is_repetition(3) else 1)
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
        sample_count = min(len(self.selfplay_buffer), self.config.phase2.batch_size * self.config.phase2.minibatches_per_epoch)
        items = [SelfPlayTransition.from_dict(item) for item in self.selfplay_buffer.sample(sample_count)]
        advantages = np.array([item.advantage for item in items], dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for idx, item in enumerate(items):
            item.advantage = float(advantages[idx])
        dataset = SelfPlayDataset(items, self.move_encoder.action_size)
        loader = DataLoader(dataset, batch_size=self.config.phase2.batch_size, shuffle=True, drop_last=False, **self._loader_kwargs())

        self.model.train()
        sums = {
            "loss_total": 0.0,
            "loss_policy": 0.0,
            "loss_search": 0.0,
            "loss_value": 0.0,
            "policy_entropy": 0.0,
            "grad_norm": 0.0,
            "tflops": 0.0,
        }
        profiled = False
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

                tflops = 0.0
                if self.config.run.profile and not profiled and batch_idx == 0 and update_epoch == 0:
                    try:
                        self.optimizer.zero_grad(set_to_none=True)
                        tflops = estimate_tflops_from_profile(lambda: forward_only(), self.device)
                        profiled = True
                    finally:
                        self.optimizer.zero_grad(set_to_none=True)

                sums["loss_total"] += float(loss.item())
                sums["loss_policy"] += float(policy_loss.item())
                sums["loss_search"] += float(search_loss.item())
                sums["loss_value"] += float(value_loss.item())
                sums["policy_entropy"] += float(entropy.item())
                sums["grad_norm"] += float(grad_norm)
                sums["tflops"] += tflops

        steps = max(self.config.phase2.update_epochs_per_cycle * max(len(loader), 1), 1)
        metrics = {key: value / steps for key, value in sums.items()}
        metrics.update(device_metrics(self.device))
        return metrics

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