"""
多模态融合模型（简单、稳定的 concat + MLP）

输入：
  vision: [B,T,35]
  audio:  [B,T,74]
  text:   [B,T,300]

做法：
  1) 各模态线性投影到 hidden_dim
  2) 对时间维做 mean pooling => [B, hidden_dim]
  3) concat => [B, 3*hidden_dim]
  4) MLP 输出
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


TaskName = Literal["emotion", "senti"]


class ConcatMLPFusionModel(nn.Module):
    def __init__(
        self,
        task: TaskName,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.task = task

        # 维度投影
        self.vision_proj = nn.Linear(35, hidden_dim)
        self.audio_proj = nn.Linear(74, hidden_dim)
        self.text_proj = nn.Linear(300, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        out_dim = 6 if task == "emotion" else 1

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, vision: torch.Tensor, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # vision/audio/text: [B,T,D]
        v = self.vision_proj(vision).mean(dim=1)  # [B,H]
        a = self.audio_proj(audio).mean(dim=1)  # [B,H]
        t = self.text_proj(text).mean(dim=1)  # [B,H]

        v = self.norm(v)
        a = self.norm(a)
        t = self.norm(t)

        fused = torch.cat([v, a, t], dim=-1)  # [B,3H]
        fused = self.dropout(fused)

        out = self.mlp(fused)  # [B,6] or [B,1]
        if self.task == "senti":
            out = out.squeeze(-1)  # [B]
        return out

