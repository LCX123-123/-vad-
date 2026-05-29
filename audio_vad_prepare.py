"""
从纯音频文件（如 RAVDESS 格式 03-02-01-01-01-01-01.wav）自动生成音频 VAD 标注 CSV。

假设文件命名遵循 RAVDESS 规则：
  MM-CC-EE-II-SS-RR-AA.wav
  - MM: Modality
  - CC: Vocal channel
  - EE: Emotion
  - II: Emotional intensity
  - SS: Statement
  - RR: Repetition
  - AA: Actor

其中 EE 情感编码：
  01 = neutral
  02 = calm
  03 = happy
  04 = sad
  05 = angry
  06 = fearful
  07 = disgust
  08 = surprised

本脚本会扫描根目录下所有 .wav 文件，解析情感类别，并依据预设的情感 -> VAD 映射生成:
  data/voice/audio_vad.csv  (默认)
列为: path,valence,arousal,dominance
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple


# 情感编号到名称（RAVDESS 风格）
EMOTION_ID2NAME: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# 简单的情感 -> VAD 映射（[-1, 1] 范围，经验近似值，可按需要微调）
EMOTION_VAD: Dict[str, Tuple[float, float, float]] = {
    "neutral": (0.0, 0.0, 0.0),
    "calm": (0.3, -0.1, 0.2),
    "happy": (0.7, 0.6, 0.4),
    "sad": (-0.6, -0.5, -0.3),
    "angry": (-0.6, 0.8, 0.5),
    "fearful": (-0.7, 0.8, -0.4),
    "disgust": (-0.6, 0.4, 0.1),
    "surprised": (0.4, 0.7, 0.2),
}


def infer_emotion_from_filename(stem: str) -> str:
    """
    从文件名（不含扩展名）解析情感名称。
    期望格式: MM-CC-EE-II-SS-RR-AA
    """
    parts = stem.split("-")
    if len(parts) < 3:
        return ""
    emo_id = parts[2]
    return EMOTION_ID2NAME.get(emo_id, "")


def main():
    parser = argparse.ArgumentParser(description="从音频文件生成 VAD 标注 CSV")
    parser.add_argument(
        "--root",
        type=str,
        default="data/voice",
        help="音频根目录（将递归搜索其中的 .wav 文件）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/voice/audio_vad.csv",
        help="输出 CSV 路径",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"音频根目录不存在: {root}")

    wav_files = list(root.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"在目录 {root} 下未找到任何 .wav 文件")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0

    for wav in wav_files:
        stem = wav.stem
        emo_name = infer_emotion_from_filename(stem)
        if not emo_name or emo_name not in EMOTION_VAD:
            skipped += 1
            continue
        v, a, d = EMOTION_VAD[emo_name]
        # 使用相对于 CSV 所在目录的相对路径，便于之后在不同机器使用
        rel_path = wav.relative_to(output_path.parent)
        rows.append((str(rel_path).replace("\\", "/"), v, a, d))

    if not rows:
        raise RuntimeError("未能从文件名中解析出任何有效情感标签，请检查命名格式是否为 RAVDESS 风格。")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "valence", "arousal", "dominance"])
        for path, v, a, d in rows:
            writer.writerow([path, f"{v:.4f}", f"{a:.4f}", f"{d:.4f}"])

    print(f"已生成音频 VAD 标注 CSV: {output_path}")
    print(f"有效样本数: {len(rows)}, 跳过样本数(无法解析情感): {skipped}")


if __name__ == "__main__":
    main()

