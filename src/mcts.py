from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import chess
import numpy as np
import torch

from .config import MCTSConfig
from .encoding import BoardEncoder, MoveEncoder, SparsePolicy
from .model import PolicyValueNet


@dataclass
class Node:
    board: chess.Board
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / max(self.visit_count, 1)


@dataclass(slots=True)
class SearchResult:
    action: int
    policy: SparsePolicy
    root_value: float
    nodes: int
    probs_dense: np.ndarray


class MCTS:
    def __init__(
        self,
        model: PolicyValueNet,
        board_encoder: BoardEncoder,
        move_encoder: MoveEncoder,
        config: MCTSConfig,
        device: torch.device,
        amp_dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder
        self.config = config
        self.device = device
        self.amp_dtype = amp_dtype

    def _evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        self.model.eval()
        state = torch.from_numpy(self.board_encoder.encode(board)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            autocast_enabled = self.device.type == "cuda" and self.amp_dtype in (torch.float16, torch.bfloat16)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=autocast_enabled):
                output = self.model(state)
            logits = output.policy_logits[0]
            value = float(output.value[0].detach().cpu().item())
        legal_mask = self.move_encoder.legal_mask(board).to(logits.device)
        priors = torch.softmax(logits.masked_fill(~legal_mask, float("-inf")), dim=-1)
        return priors.detach().cpu().to(torch.float32).numpy(), value

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        if not board.is_game_over(claim_draw=True):
            raise ValueError("_terminal_value expects a terminal board.")
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    def _expand(self, node: Node) -> float:
        if node.board.is_game_over(claim_draw=True):
            node.expanded = True
            return self._terminal_value(node.board)

        priors, value = self._evaluate(node.board)
        legal_moves = list(node.board.legal_moves)
        for move in legal_moves:
            action = self.move_encoder.move_to_index(move)
            child_board = node.board.copy(stack=False)
            child_board.push(move)
            node.children[action] = Node(board=child_board, prior=float(priors[action]))
        node.expanded = True
        return value

    def _add_root_noise(self, node: Node) -> None:
        if not self.config.root_noise or not node.children:
            return
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
        for idx, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1.0 - self.config.dirichlet_eps) * child.prior + self.config.dirichlet_eps * float(noise[idx])

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        parent_visits = math.sqrt(max(node.visit_count, 1))
        best_score = -float("inf")
        best_action = -1
        best_child: Optional[Node] = None

        for action, child in node.children.items():
            q = -child.q_value
            u = self.config.c_puct * child.prior * parent_visits / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("MCTS failed to select a child.")
        return best_action, best_child

    def run(self, board: chess.Board, ply: int) -> SearchResult:
        root = Node(board=board.copy(stack=False), prior=1.0)
        self._expand(root)
        self._add_root_noise(root)
        nodes = 0

        for _ in range(self.config.simulations):
            path = [root]
            node = root
            while node.expanded and node.children:
                _, node = self._select_child(node)
                path.append(node)
            if not node.expanded:
                value = self._expand(node)
            else:
                value = self._terminal_value(node.board)

            for ancestor in reversed(path):
                ancestor.visit_count += 1
                ancestor.value_sum += value
                value = -value
            nodes += 1

        actions = np.array(list(root.children.keys()), dtype=np.int64)
        counts = np.array([root.children[action].visit_count for action in actions], dtype=np.float32)
        if counts.sum() <= 0:
            counts = np.ones_like(counts)
        temperature = self._temperature_for_ply(ply)
        if temperature <= 1e-6:
            best_idx = int(np.argmax(counts))
            probs = np.zeros_like(counts)
            probs[best_idx] = 1.0
        else:
            tempered = np.power(counts, 1.0 / temperature)
            probs = tempered / np.sum(tempered)
        chosen_local_idx = int(np.random.choice(len(actions), p=probs))
        chosen_action = int(actions[chosen_local_idx])
        dense = np.zeros(self.move_encoder.action_size, dtype=np.float32)
        dense[actions] = probs.astype(np.float32)
        return SearchResult(
            action=chosen_action,
            policy=SparsePolicy(indices=actions, probs=probs.astype(np.float32)),
            root_value=root.q_value,
            nodes=nodes,
            probs_dense=dense,
        )

    def _temperature_for_ply(self, ply: int) -> float:
        start = self.config.temperature_start
        end = self.config.temperature_end
        decay = max(self.config.temperature_decay_plies, 1)
        ratio = min(max(ply / decay, 0.0), 1.0)
        return start + ratio * (end - start)