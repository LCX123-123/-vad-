"""
MOSEI 多模态融合预测与解释（基于你已有的 pkl 预提取特征）

数据来源（你当前 cmumosei 目录）：
  - mosei_senti_data.pkl: labels 为回归标量（[-3,3]）
  - mosei_emotion_aligned_60.pkl: labels 为6维多标签二分类（0/1）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from mosei_multimodal_dataset import MoseiMultimodalDataset, TaskName
from mosei_fusion_model import ConcatMLPFusionModel

PredictTask = Literal["senti", "emotion"]


def interpret_senti(score: float) -> str:
    # score 近似范围[-3,3]，这里用经验阈值做“积极/中性/消极”
    if score >= 1.0:
        return "非常积极"
    if score >= 0.2:
        return "积极"
    if score > -0.2:
        return "中性"
    if score > -1.0:
        return "消极"
    return "非常消极"


def interpret_emotion(probs: np.ndarray, threshold: float = 0.5, names: Optional[List[str]] = None) -> str:
    """
    probs: [6] 每个维度的概率（sigmoid输出）
    """
    if probs.shape[0] != 6:
        raise ValueError(f"emotion probs expects shape(6,), got {probs.shape}")
    if names is None:
        names = [f"dim{i}" for i in range(6)]
    active = []
    for i, p in enumerate(probs.tolist()):
        if p >= threshold:
            active.append(f"{names[i]}({p:.2f})")
    if not active:
        return "未检测到明显情感维度（低于阈值）"
    return " | ".join(active)


def load_fusion_model(
    task: TaskName,
    checkpoint_path: str,
    device: str,
    hidden_dim: int = 256,
    dropout: float = 0.2,
) -> ConcatMLPFusionModel:
    model = ConcatMLPFusionModel(task=task, hidden_dim=hidden_dim, dropout=dropout)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 兼容不同保存格式
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_on_indices(
    task: TaskName,
    model: ConcatMLPFusionModel,
    dataset: MoseiMultimodalDataset,
    indices: List[int],
    device: str,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_preds = []
    all_labels = []
    for batch in loader:
        vision = batch["vision"].to(device)
        audio = batch["audio"].to(device)
        text = batch["text"].to(device)
        labels = batch["labels"].to(device)
        out = model(vision, audio, text)
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return preds, labels


def predict_single_by_index(
    task: TaskName,
    checkpoint_path: str,
    dataset_pkl_path: str,
    split: Literal["train", "valid", "test"],
    index: int,
    device: str,
    hidden_dim: int = 256,
    dropout: float = 0.2,
) -> Dict:
    dataset = MoseiMultimodalDataset(dataset_pkl_path, task=task, split=split)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index out of range: {index}, dataset_len={len(dataset)}")
    model = load_fusion_model(task=task, checkpoint_path=checkpoint_path, device=device, hidden_dim=hidden_dim, dropout=dropout)

    preds, labels = predict_on_indices(task, model, dataset, [index], device=device, batch_size=1)
    # preds: [1,6] logits or [1]
    result: Dict = {"index": index, "split": split}

    if task == "emotion":
        logits = preds[0]  # [6]
        probs = 1 / (1 + np.exp(-logits))
        # labels shape: [1,6]
        result["probs"] = {f"dim{i}": float(probs[i]) for i in range(6)}
        result["interpretation"] = interpret_emotion(probs, threshold=0.5)
        result["gt_labels"] = {f"dim{i}": float(labels.reshape(-1, 6)[0, i]) for i in range(6)}
    else:
        score = float(preds.reshape(-1)[0])
        result["score"] = score
        result["interpretation"] = interpret_senti(score)
        result["gt_label"] = float(labels.reshape(-1)[0])

    return result


def export_split_predictions(
    task: TaskName,
    checkpoint_path: str,
    dataset_pkl_path: str,
    split: Literal["train", "valid", "test"],
    device: str,
    out_csv: str,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    batch_size: int = 16,
    limit: Optional[int] = None,
) -> str:
    """导出某个 split 的预测结果到 CSV（按 index 输出）。"""
    import pandas as pd

    dataset = MoseiMultimodalDataset(dataset_pkl_path, task=task, split=split)
    indices = list(range(len(dataset)))
    if limit is not None:
        indices = indices[: int(limit)]

    model = load_fusion_model(
        task=task,
        checkpoint_path=checkpoint_path,
        device=device,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    pred_rows = []
    cursor = 0
    for batch in loader:
        bsz = batch["vision"].shape[0]
        vision = batch["vision"].to(device)
        audio = batch["audio"].to(device)
        text = batch["text"].to(device)
        labels = batch["labels"].detach().cpu().numpy()

        with torch.no_grad():
            out = model(vision, audio, text)

        out_np = out.detach().cpu().numpy()
        for i in range(bsz):
            global_idx = indices[cursor + i]
            if task == "emotion":
                logits = out_np[i]  # [6]
                probs = 1 / (1 + np.exp(-logits))
                row = {
                    "index": int(global_idx),
                }
                for d in range(6):
                    row[f"pred_dim{d}"] = float(probs[d])
                    row[f"gt_dim{d}"] = int(float(labels[i].reshape(-1)[d]))
                pred_rows.append(row)
            else:
                row = {
                    "index": int(global_idx),
                    "pred_score": float(out_np.reshape(-1)[cursor + i]),
                    "gt_score": float(labels.reshape(-1)[cursor + i]),
                }
                pred_rows.append(row)
        cursor += bsz

    df = pd.DataFrame(pred_rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return str(out_path)

