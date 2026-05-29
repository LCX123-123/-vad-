"""
VAD（三维回归）评估与落盘报告工具。

目标：
- 兼容图像 VAD（ResNet18 回归头）与音频 VAD（Wav2Vec2 回归头）等输出形状为 [N,3] 的回归模型
- 输出与文本模态类似的“可落盘评估报告”：json + txt（便于论文引用与截图）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import torch
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DIMS = ["valence", "arousal", "dominance"]


@dataclass
class VADReport:
    metrics: Dict
    predictions: np.ndarray  # [N,3]
    labels: np.ndarray  # [N,3]


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c) or np.isinf(c):
        return 0.0
    return float(c)


def calculate_vad_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    dims: Optional[List[str]] = None,
) -> Dict:
    dims = dims or DEFAULT_DIMS
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    if p.ndim != 2 or y.ndim != 2:
        raise ValueError(f"predictions/labels 必须是二维数组，得到 p={p.shape}, y={y.shape}")
    if p.shape != y.shape:
        raise ValueError(f"predictions/labels 形状不一致：p={p.shape}, y={y.shape}")
    if p.shape[1] != len(dims):
        raise ValueError(f"维度数不匹配：p.shape[1]={p.shape[1]} vs dims={len(dims)}")

    results: Dict = {"overall": {}, "dimensions": {}, "correlations": {}}

    results["overall"] = {
        "mse": float(mean_squared_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "explained_variance": float(explained_variance_score(y, p)),
        # sklearn.max_error 不支持 multioutput，这里用逐元素误差的全局最大值
        "max_error": float(np.max(np.abs(y - p))),
        "pearson_r_mean": float(np.mean([_safe_pearsonr(p[:, i], y[:, i]) for i in range(len(dims))])),
    }

    for i, d in enumerate(dims):
        yi = y[:, i]
        pi = p[:, i]
        results["dimensions"][d] = {
            "mse": float(mean_squared_error(yi, pi)),
            "rmse": float(np.sqrt(mean_squared_error(yi, pi))),
            "mae": float(mean_absolute_error(yi, pi)),
            "r2": float(r2_score(yi, pi)),
            "explained_variance": float(explained_variance_score(yi, pi)),
            "max_error": float(np.max(np.abs(yi - pi))),
            "pearson_r": float(_safe_pearsonr(pi, yi)),
        }

    results["correlations"] = {
        "predictions": {"matrix": np.corrcoef(p.T).tolist(), "dimensions": dims},
        "labels": {"matrix": np.corrcoef(y.T).tolist(), "dimensions": dims},
        "prediction_accuracy": {
            "correlations": [float(_safe_pearsonr(p[:, i], y[:, i])) for i in range(len(dims))],
            "dimensions": dims,
        },
    }

    return results


@torch.no_grad()
def run_regression_inference(
    model: torch.nn.Module,
    data_loader,
    device: str,
    batch_to_xy,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    通用推理循环：
    - batch_to_xy(batch) -> (x, y)，其中 x 会被送入 model，y 作为标签收集
    - model(x) 应输出 [B,3]（或可 reshape 成 [B,3]）
    """
    model.eval()
    model.to(device)
    all_p: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for batch in data_loader:
        x, y = batch_to_xy(batch)
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        all_p.append(pred.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    p = np.concatenate(all_p, axis=0)
    y = np.concatenate(all_y, axis=0)
    return p, y


def save_vad_report(
    report_dir: str | Path,
    metrics: Dict,
    extra: Optional[Dict] = None,
    json_name: str = "evaluation_results.json",
    txt_name: str = "evaluation_report.txt",
) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {"metrics": metrics}
    if extra:
        payload["extra"] = extra

    json_path = report_dir / json_name
    txt_path = report_dir / txt_name

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 生成一个“适合截图”的简洁文本报告
    dims = metrics.get("correlations", {}).get("prediction_accuracy", {}).get("dimensions", DEFAULT_DIMS)
    lines: List[str] = []
    lines.append("VAD 回归评估报告（Test）")
    lines.append("")
    overall = metrics.get("overall", {})
    lines.append("【总体指标】")
    lines.append(f"mse={overall.get('mse', 0.0):.6f}  rmse={overall.get('rmse', 0.0):.6f}  mae={overall.get('mae', 0.0):.6f}")
    lines.append(f"r2={overall.get('r2', 0.0):.6f}  explained_variance={overall.get('explained_variance', 0.0):.6f}")
    lines.append(f"max_error={overall.get('max_error', 0.0):.6f}  pearson_r_mean={overall.get('pearson_r_mean', 0.0):.6f}")
    lines.append("")
    lines.append("【分维度指标】")
    for d in dims:
        dm = metrics.get("dimensions", {}).get(d, {})
        lines.append(
            f"{d}: mse={dm.get('mse', 0.0):.6f}  mae={dm.get('mae', 0.0):.6f}  r2={dm.get('r2', 0.0):.6f}  pearson_r={dm.get('pearson_r', 0.0):.6f}"
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return report_dir

