from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(slots=True)
class ProjectConfig:
    name: str = "chess_hybrid_rl"
    seed: int = 42
    output_dir: str = "./chess_hybrid_rl_runs"
    verbosity: str = "info"
    device: str = "auto"
    dtype: str = "bf16"
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    allow_tf32: bool = True


@dataclass(slots=True)
class HardwareConfig:
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    cudnn_benchmark: bool = True
    gradient_accumulation_steps: int = 1


@dataclass(slots=True)
class RunConfig:
    stage: str = "full"
    continue_from: str = ""
    clear_cache: bool = False
    safe_save_on_error: bool = True
    profile: bool = True


@dataclass(slots=True)
class TeacherConfig:
    enabled: bool = True
    engine_path: str = "stockfish"
    threads: int = 8
    hash_mb: int = 2048
    movetime_ms: int = 30
    depth: Optional[int] = None
    multipv: int = 1
    cp_to_value_scale: float = 600.0
    policy_temperature: float = 80.0


@dataclass(slots=True)
class Phase1Config:
    epochs: int = 8
    batch_size: int = 1024
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    label_smoothing: float = 0.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    entropy_weight: float = 1.0e-3
    teacher_samples_per_epoch: int = 12000
    teacher_games_per_epoch: int = 32
    max_teacher_game_plies: int = 100
    robust_distillation: bool = False
    opening_noise_min_plies: int = 6
    opening_noise_max_plies: int = 18
    move_injection_best_prob: float = 0.60
    move_injection_topk_prob: float = 0.25
    move_injection_random_prob: float = 0.15
    move_injection_topk: int = 4


@dataclass(slots=True)
class Phase2Config:
    epochs: int = 40
    batch_size: int = 1024
    lr: float = 1.5e-4
    weight_decay: float = 1.0e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    search_loss_weight: float = 1.0
    entropy_weight: float = 5.0e-4
    replay_buffer_capacity: int = 200000
    selfplay_games_per_epoch: int = 32
    max_game_plies: int = 200
    update_epochs_per_cycle: int = 4
    minibatches_per_epoch: int = 32


@dataclass(slots=True)
class CurriculumConfig:
    overlap_start_epoch: int = 5
    overlap_span_epochs: int = 8
    teacher_ratio_start: float = 1.0
    teacher_ratio_end: float = 0.0
    kl_anchor_weight_start: float = 0.02
    kl_anchor_weight_end: float = 0.0


@dataclass(slots=True)
class RewardConfig:
    gamma: float = 0.997
    cp_delta_scale: float = 120.0
    w_cp: float = 0.60
    w_match: float = 0.25
    w_terminal: float = 0.15
    w_stall: float = 0.02
    draw_penalty: bool = True
    draw_base: float = 0.15
    draw_tau: float = 80.0
    repetition_penalty: float = 0.03


@dataclass(slots=True)
class GAEConfig:
    gamma: float = 0.997
    lambda_min: float = 0.80
    lambda_max: float = 0.97
    lambda_init: float = 0.92
    ema_beta: float = 0.90
    k_noise: float = 0.12
    k_value_var: float = 0.05


@dataclass(slots=True)
class MCTSConfig:
    simulations: int = 128
    c_puct: float = 1.75
    dirichlet_alpha: float = 0.30
    dirichlet_eps: float = 0.25
    temperature_start: float = 1.00
    temperature_end: float = 0.10
    temperature_decay_plies: int = 20
    root_noise: bool = True


@dataclass(slots=True)
class ModelConfig:
    input_planes: int = 18
    channels: int = 128
    num_blocks: int = 10
    value_head_hidden: int = 256
    dropout: float = 0.0
    action_size: int = 20480


@dataclass(slots=True)
class EvalConfig:
    every_n_epochs: int = 5
    arena_games: int = 8
    opponent_elo: float = 2900.0
    baseline_movetime_ms: int = 20
    max_game_plies: int = 160


@dataclass(slots=True)
class LoggingConfig:
    log_every_steps: int = 20
    checkpoint_every_epochs: int = 1
    emergency_checkpoint_every_minutes: int = 15
    tensorboard: bool = True
    jsonl: bool = True
    profile_every_steps: int = 200


@dataclass(slots=True)
class TrainConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    run: RunConfig = field(default_factory=RunConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    gae: GAEConfig = field(default_factory=GAEConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def output_dir(self) -> Path:
        return Path(self.project.output_dir).expanduser().resolve()


_SECTION_TYPES = {
    "project": ProjectConfig,
    "hardware": HardwareConfig,
    "run": RunConfig,
    "teacher": TeacherConfig,
    "phase1": Phase1Config,
    "phase2": Phase2Config,
    "curriculum": CurriculumConfig,
    "reward": RewardConfig,
    "gae": GAEConfig,
    "mcts": MCTSConfig,
    "model": ModelConfig,
    "eval": EvalConfig,
    "logging": LoggingConfig,
}


def _construct_section(cls: Any, raw: Optional[Dict[str, Any]]) -> Any:
    raw = raw or {}
    return cls(**raw)


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    kwargs = {name: _construct_section(cls, raw.get(name)) for name, cls in _SECTION_TYPES.items()}
    return TrainConfig(**kwargs)


def dump_config(config: TrainConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
