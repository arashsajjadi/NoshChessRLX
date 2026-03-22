from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import chess
import chess.engine
import numpy as np

from .config import TeacherConfig
from .encoding import MoveEncoder, SparsePolicy


@dataclass(slots=True)
class TeacherAnalysis:
    best_move: chess.Move
    value: float
    cp: float
    nodes: int
    nps: int
    policy: SparsePolicy


class StockfishTeacher:
    def __init__(self, config: TeacherConfig, move_encoder: MoveEncoder) -> None:
        self.config = config
        self.move_encoder = move_encoder
        self.engine = chess.engine.SimpleEngine.popen_uci(config.engine_path)
        self.engine.configure({"Threads": config.threads, "Hash": config.hash_mb})

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self) -> "StockfishTeacher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _limit(self) -> chess.engine.Limit:
        if self.config.depth is not None:
            return chess.engine.Limit(depth=self.config.depth)
        return chess.engine.Limit(time=max(self.config.movetime_ms, 1) / 1000.0)

    def _score_to_cp(self, score: chess.engine.PovScore, turn: chess.Color) -> float:
        relative = score.pov(turn).score(mate_score=100000)
        return float(relative if relative is not None else 0.0)

    def _softmax(self, values: List[float]) -> np.ndarray:
        if not values:
            return np.zeros(0, dtype=np.float32)
        array = np.array(values, dtype=np.float32)
        array = array / max(self.config.policy_temperature, 1e-6)
        array = array - np.max(array)
        probs = np.exp(array)
        probs /= np.sum(probs)
        return probs.astype(np.float32)

    def analyze(self, board: chess.Board) -> TeacherAnalysis:
        info = self.engine.analyse(board, self._limit(), multipv=self.config.multipv)
        rows = info if isinstance(info, list) else [info]
        moves: List[chess.Move] = []
        cps: List[float] = []
        nodes = 0
        nps = 0

        for row in rows:
            pv = row.get("pv", [])
            if not pv:
                continue
            move = pv[0]
            moves.append(move)
            cps.append(self._score_to_cp(row["score"], board.turn))
            nodes = max(nodes, int(row.get("nodes", 0)))
            nps = max(nps, int(row.get("nps", 0)))

        if not moves:
            best_move = next(iter(board.legal_moves))
            cp = 0.0
            probs = np.array([1.0], dtype=np.float32)
            indices = np.array([self.move_encoder.move_to_index(best_move)], dtype=np.int64)
        else:
            best_move = moves[0]
            cp = cps[0]
            probs = self._softmax(cps)
            indices = np.array([self.move_encoder.move_to_index(move) for move in moves], dtype=np.int64)

        value = math.tanh(cp / max(self.config.cp_to_value_scale, 1e-6))
        return TeacherAnalysis(
            best_move=best_move,
            value=float(value),
            cp=float(cp),
            nodes=nodes,
            nps=nps,
            policy=SparsePolicy(indices=indices, probs=probs),
        )