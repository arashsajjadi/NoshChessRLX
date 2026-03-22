from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .encoding import SparsePolicy


@dataclass(slots=True)
class TeacherSample:
    state: np.ndarray
    policy_indices: np.ndarray
    policy_probs: np.ndarray
    value_target: float

    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "policy_indices": self.policy_indices,
            "policy_probs": self.policy_probs,
            "value_target": self.value_target,
        }

    @staticmethod
    def from_dict(data: Dict) -> "TeacherSample":
        return TeacherSample(
            state=data["state"],
            policy_indices=data["policy_indices"],
            policy_probs=data["policy_probs"],
            value_target=float(data["value_target"]),
        )


@dataclass(slots=True)
class SelfPlayTransition:
    state: np.ndarray
    action: int
    search_policy_indices: np.ndarray
    search_policy_probs: np.ndarray
    reward: float
    value_pred: float
    log_prob: float
    done: bool
    next_state: np.ndarray | None
    player_sign: float
    return_target: float = 0.0
    advantage: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "action": self.action,
            "search_policy_indices": self.search_policy_indices,
            "search_policy_probs": self.search_policy_probs,
            "reward": self.reward,
            "value_pred": self.value_pred,
            "log_prob": self.log_prob,
            "done": self.done,
            "next_state": self.next_state,
            "player_sign": self.player_sign,
            "return_target": self.return_target,
            "advantage": self.advantage,
        }

    @staticmethod
    def from_dict(data: Dict) -> "SelfPlayTransition":
        return SelfPlayTransition(
            state=data["state"],
            action=int(data["action"]),
            search_policy_indices=data["search_policy_indices"],
            search_policy_probs=data["search_policy_probs"],
            reward=float(data["reward"]),
            value_pred=float(data["value_pred"]),
            log_prob=float(data["log_prob"]),
            done=bool(data["done"]),
            next_state=data["next_state"],
            player_sign=float(data["player_sign"]),
            return_target=float(data.get("return_target", 0.0)),
            advantage=float(data.get("advantage", 0.0)),
        )


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: Deque[Dict] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, item: Dict) -> None:
        self.data.append(item)

    def extend(self, items: Iterable[Dict]) -> None:
        self.data.extend(items)

    def sample(self, batch_size: int) -> List[Dict]:
        batch_size = min(batch_size, len(self.data))
        return random.sample(list(self.data), batch_size)

    def state_dict(self) -> Dict:
        return {"capacity": self.capacity, "data": list(self.data)}

    def load_state_dict(self, state: Dict) -> None:
        self.capacity = int(state["capacity"])
        self.data = deque(state["data"], maxlen=self.capacity)


class TeacherDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, items: Sequence[TeacherSample], action_size: int) -> None:
        self.items = list(items)
        self.action_size = action_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        policy = np.zeros(self.action_size, dtype=np.float32)
        policy[item.policy_indices] = item.policy_probs
        return {
            "states": torch.from_numpy(item.state),
            "policy_targets": torch.from_numpy(policy),
            "value_targets": torch.tensor(item.value_target, dtype=torch.float32),
        }


class SelfPlayDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, items: Sequence[SelfPlayTransition], action_size: int) -> None:
        self.items = list(items)
        self.action_size = action_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        policy = np.zeros(self.action_size, dtype=np.float32)
        policy[item.search_policy_indices] = item.search_policy_probs
        return {
            "states": torch.from_numpy(item.state),
            "actions": torch.tensor(item.action, dtype=torch.long),
            "search_policy": torch.from_numpy(policy),
            "returns": torch.tensor(item.return_target, dtype=torch.float32),
            "advantages": torch.tensor(item.advantage, dtype=torch.float32),
            "old_log_probs": torch.tensor(item.log_prob, dtype=torch.float32),
        }