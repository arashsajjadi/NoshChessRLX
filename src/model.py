from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + residual)
        return x


@dataclass(slots=True)
class NetworkOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor


class PolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = config.channels
        self.action_size = config.action_size
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(*[ResidualBlock(channels) for _ in range(config.num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, config.action_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, config.value_head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_head_hidden, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> NetworkOutput:
        x = self.stem(x)
        x = self.backbone(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return NetworkOutput(policy_logits=logits, value=value)