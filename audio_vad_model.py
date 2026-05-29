"""
音频 VAD 回归模型
使用 torchaudio 的预训练特征（例如 Wav2Vec2）+ 全连接层输出 (valence, arousal, dominance)。
"""

from typing import Dict

import torch
import torch.nn as nn
import torchaudio


class AudioVADRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "wav2vec2_base",
        num_outputs: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        if backbone_name == "wav2vec2_base":
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
        else:
            raise ValueError(f"不支持的音频骨干网络: {backbone_name}")

        self.backbone = bundle.get_model()

        # 通过一次前向推理自动推断特征维度，避免依赖内部实现细节
        with torch.no_grad():
            dummy_wav = torch.zeros(1, int(bundle.sample_rate * 1.0))  # 1 秒静音
            feats, _ = self.backbone.extract_features(dummy_wav)
            feat_dim = feats[-1].shape[-1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: [B, 1, T]
        Returns:
            vad: [B, 3] in [-1,1] range (通过 tanh 约束可选)
        """
        # Wav2Vec2 期望 [B, T]
        x = waveforms.squeeze(1)
        feat, _ = self.backbone.extract_features(x)  # list of [B, T', C]
        feat = feat[-1].mean(dim=1)  # [B, C]
        out = self.head(feat)
        return torch.tanh(out)


def interpret_audio_vad(vad: Dict[str, float]) -> str:
    v, a, d = vad.get("valence", 0.0), vad.get("arousal", 0.0), vad.get("dominance", 0.0)

    # 更细的阈值划分
    if v > 0.6:
        v_str = "非常积极"
    elif v > 0.2:
        v_str = "略微积极"
    elif v > -0.2:
        v_str = "中性"
    elif v > -0.6:
        v_str = "略微消极"
    else:
        v_str = "非常消极"

    if a > 0.6:
        a_str = "高唤醒"
    elif a > 0.2:
        a_str = "中等唤醒"
    elif a > -0.2:
        a_str = "平静"
    else:
        a_str = "低唤醒"

    if d > 0.6:
        d_str = "高支配性"
    elif d > 0.2:
        d_str = "中等支配性"
    elif d > -0.2:
        d_str = "中性支配"
    else:
        d_str = "低支配性"

    return f"{v_str} | {a_str} | {d_str}"

