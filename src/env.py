from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import chess

from .config import RewardConfig


@dataclass(slots=True)
class TransitionReward:
    reward: float
    terminal: bool
    result: float
    is_draw: bool


class RewardShaper:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    @staticmethod
    def cp_to_unit(cp: float, scale: float) -> float:
        return math.tanh(cp / max(scale, 1e-6))

    def draw_reward(self, ply: int) -> float:
        if not self.config.draw_penalty:
            return 0.0
        return -self.config.draw_base * math.exp(-ply / max(self.config.draw_tau, 1e-6))

    def dense_reward(
        self,
        cp_before: float,
        cp_after: float,
        action_matches_teacher: bool,
        repetition_count: int,
    ) -> float:
        cp_term = self.config.w_cp * math.tanh((cp_after - cp_before) / max(self.config.cp_delta_scale, 1e-6))
        match_term = self.config.w_match if action_matches_teacher else 0.0
        stall_term = -self.config.w_stall * max(repetition_count - 1, 0)
        if repetition_count >= 3:
            stall_term -= self.config.repetition_penalty
        return cp_term + match_term + stall_term

    def terminal_reward(self, board: chess.Board, actor_color: chess.Color, ply: int) -> TransitionReward:
        if not board.is_game_over(claim_draw=True):
            return TransitionReward(reward=0.0, terminal=False, result=0.0, is_draw=False)

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            reward = self.draw_reward(ply)
            return TransitionReward(reward=reward, terminal=True, result=0.0, is_draw=True)

        result = 1.0 if outcome.winner == actor_color else -1.0
        reward = self.config.w_terminal * result
        return TransitionReward(reward=reward, terminal=True, result=result, is_draw=False)


class ChessEnv:
    def __init__(self, reward_config: RewardConfig) -> None:
        self.board = chess.Board()
        self.reward_shaper = RewardShaper(reward_config)

    def reset(self, fen: Optional[str] = None) -> chess.Board:
        self.board = chess.Board(fen) if fen else chess.Board()
        return self.board

    def copy_board(self) -> chess.Board:
        return self.board.copy(stack=False)

    def push(self, move: chess.Move) -> None:
        self.board.push(move)

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=True)