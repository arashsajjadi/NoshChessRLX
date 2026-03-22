from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import chess
import numpy as np
import torch


PROMOTION_TO_INDEX = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}
INDEX_TO_PROMOTION = {value: key for key, value in PROMOTION_TO_INDEX.items()}
PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


@dataclass(slots=True)
class SparsePolicy:
    indices: np.ndarray
    probs: np.ndarray

    def to_dense(self, size: int) -> np.ndarray:
        dense = np.zeros(size, dtype=np.float32)
        dense[self.indices] = self.probs
        return dense


class MoveEncoder:
    """
    Maps UCI moves to a fixed action vocabulary.

    Layout: ((from_square * 64) + to_square) * 5 + promotion_bucket
    promotion_bucket: 0 none, 1 queen, 2 rook, 3 bishop, 4 knight
    """

    def __init__(self) -> None:
        self.action_size = 64 * 64 * 5

    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        promo_idx = PROMOTION_TO_INDEX[move.promotion]
        return ((move.from_square * 64) + move.to_square) * 5 + promo_idx

    @staticmethod
    def index_to_move(index: int) -> chess.Move:
        base, promo_idx = divmod(index, 5)
        from_square, to_square = divmod(base, 64)
        promotion = INDEX_TO_PROMOTION[promo_idx]
        return chess.Move(from_square, to_square, promotion=promotion)

    def legal_indices(self, board: chess.Board) -> List[int]:
        return [self.move_to_index(move) for move in board.legal_moves]

    def legal_mask(self, board: chess.Board) -> torch.Tensor:
        mask = torch.zeros(self.action_size, dtype=torch.bool)
        legal = self.legal_indices(board)
        if legal:
            mask[legal] = True
        return mask


class BoardEncoder:
    """
    18 x 8 x 8 tensor.
    0..11 piece planes
    12 side to move
    13..16 castling rights
    17 halfmove clock normalized
    """

    def __init__(self) -> None:
        self.num_planes = 18

    def encode(self, board: chess.Board) -> np.ndarray:
        planes = np.zeros((self.num_planes, 8, 8), dtype=np.float32)

        for square, piece in board.piece_map().items():
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            planes[PIECE_TO_PLANE[(piece.piece_type, piece.color)], rank, file] = 1.0

        planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        planes[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)

        if board.turn == chess.BLACK:
            planes = np.flip(planes, axis=2).copy()
            planes = np.flip(planes, axis=1).copy()
            white_planes = planes[0:6].copy()
            black_planes = planes[6:12].copy()
            planes[0:6] = black_planes
            planes[6:12] = white_planes
            castle = planes[13:17].copy()
            planes[13] = castle[2]
            planes[14] = castle[3]
            planes[15] = castle[0]
            planes[16] = castle[1]

        return planes

    def batch(self, boards: List[chess.Board]) -> torch.Tensor:
        array = np.stack([self.encode(board) for board in boards], axis=0)
        return torch.from_numpy(array)


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    return torch.softmax(masked, dim=dim)