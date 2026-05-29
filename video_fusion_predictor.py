import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torchaudio

from emotion_dimension_model import create_emotion_dimension_model


def _run_ffmpeg(args: List[str], ffmpeg_path: str = "ffmpeg") -> None:
    # 让报错信息更明确
    try:
        proc = subprocess.run([ffmpeg_path] + args, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"找不到 ffmpeg 可执行文件: {ffmpeg_path}\n"
            "请安装 ffmpeg 并加入系统 PATH，或在命令里使用 --ffmpeg_path 指定 ffmpeg.exe 的完整路径。"
        ) from e
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg 执行失败:\n"
            f"cmd: {' '.join([ffmpeg_path] + args)}\n"
            f"stdout: {proc.stdout[:2000]}\n"
            f"stderr: {proc.stderr[:4000]}"
        )


def extract_audio_16k(video_path: str, out_wav: str, mono: bool = True, ffmpeg_path: str = "ffmpeg") -> None:
    # -vn: no video
    # -ac 1: 单声道
    # -ar 16000: 16kHz
    args = [
        "-y",
        "-i",
        video_path,
        "-vn",
    ]
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", "16000", out_wav]
    _run_ffmpeg(args, ffmpeg_path=ffmpeg_path)


def extract_frames(video_path: str, out_dir: str, fps: float = 1.0, max_frames: int = 64, ffmpeg_path: str = "ffmpeg") -> List[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # frame_%06d.jpg
    # fps 过大耗时，max_frames 用于后续截断
    out_pattern = str(Path(out_dir) / "frame_%06d.jpg")
    args = ["-y", "-i", video_path, "-vf", f"fps={fps}", out_pattern]
    _run_ffmpeg(args, ffmpeg_path=ffmpeg_path)

    frames = sorted(Path(out_dir).glob("frame_*.jpg"))
    frames = frames[:max_frames]
    return [str(p) for p in frames]


def try_extract_subtitles(video_path: str, out_srt: str, ffmpeg_path: str = "ffmpeg") -> bool:
    """
    尝试从视频中抽取第一个字幕流为 srt。
    如果没有字幕流，ffmpeg 会失败并抛异常，这里捕获返回 False。
    """
    try:
        Path(out_srt).parent.mkdir(parents=True, exist_ok=True)
        args = ["-y", "-i", video_path, "-map", "0:s:0", out_srt]
        _run_ffmpeg(args, ffmpeg_path=ffmpeg_path)
        return Path(out_srt).exists() and Path(out_srt).stat().st_size > 0
    except Exception:
        return False


def read_srt_as_text(srt_path: str) -> str:
    """
    粗略把 srt 转成文本（去掉时间戳/序号/空行）。
    """
    p = Path(srt_path)
    if not p.exists():
        return ""
    raw = p.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # 跳过纯数字序号
        if re.fullmatch(r"\d+", line):
            continue
        # 跳过时间戳行，如 00:00:01,000 --> 00:00:03,000
        if "-->" in line:
            continue
        # 清理一些标签
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)
    text = " ".join(lines)
    # 过长文本不利于 tokenization，这里做轻度截断
    return text[:20000]


def interpret_vad_common(vad: Dict[str, float]) -> str:
    v, a, d = vad.get("valence", 0.0), vad.get("arousal", 0.0), vad.get("dominance", 0.0)

    # 更细阈值，避免像你之前遇到的“0.0283 却被归为积极”
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


def _load_text_vad_model(text_model_path: str, device: str, emotion_dimensions=None):
    emotion_dimensions = emotion_dimensions or ["valence", "arousal", "dominance"]
    # 与你现有训练一致：VAD dimension model
    model = create_emotion_dimension_model(
        model_name="bert-base-uncased",
        emotion_dimensions=emotion_dimensions,
        model_type="dimension",
        hidden_dims=[768, 256, 64],
        dropout_rate=0.3,
        freeze_bert=False,
        max_length=512,
        use_attention_pooling=True,
    )
    ckpt = torch.load(text_model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, emotion_dimensions


def _load_image_vad_model(image_model_path: str, device: str):
    from image_fer2013_vad import VADRegressor

    model = VADRegressor(pretrained=False)
    ckpt = torch.load(image_model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _load_audio_vad_model(audio_model_path: str, device: str):
    from audio_vad_model import AudioVADRegressor

    model = AudioVADRegressor()
    ckpt = torch.load(audio_model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _preprocess_image_for_vad(frame_jpg_path: str, device: str) -> torch.Tensor:
    from PIL import Image
    import torch.nn.functional as F

    img = Image.open(frame_jpg_path).convert("L")
    img = img.resize((48, 48), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    img_tensor = torch.from_numpy(arr).unsqueeze(0).float() / 255.0  # (1,48,48)
    img_tensor = F.interpolate(
        img_tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)
    img_tensor = img_tensor.repeat(3, 1, 1)  # (3,224,224)
    return img_tensor.unsqueeze(0).to(device)  # (1,3,224,224)


@torch.no_grad()
def predict_vad_from_video(
    video_path: str,
    device: str,
    text_model_path: str,
    image_model_path: str,
    audio_model_path: str,
    ffmpeg_path: str = "ffmpeg",
    video_fps: float = 1.0,
    max_frames: int = 64,
    audio_chunk_seconds: float = 6.0,
    text_weight: float = 1.0,
    audio_weight: float = 1.0,
    image_weight: float = 1.0,
    subtitle_srt_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    video_path = str(video_path)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    tmp_dir = tempfile.mkdtemp(prefix="video_fusion_")
    try:
        # 1) extract: audio + frames + (optional) subtitles
        wav_path = str(Path(tmp_dir) / "audio_16k.wav")
        extract_audio_16k(video_path, wav_path, mono=True, ffmpeg_path=ffmpeg_path)

        frames_dir = str(Path(tmp_dir) / "frames")
        frame_paths = extract_frames(
            video_path, frames_dir, fps=video_fps, max_frames=max_frames, ffmpeg_path=ffmpeg_path
        )

        srt_path = str(Path(tmp_dir) / "subtitles.srt")
        subtitle_text = ""
        if subtitle_srt_path:
            subtitle_text = read_srt_as_text(subtitle_srt_path)
        else:
            # try auto extract
            if try_extract_subtitles(video_path, srt_path, ffmpeg_path=ffmpeg_path):
                subtitle_text = read_srt_as_text(srt_path)

        # 2) load models
        text_model, emotion_dimensions = _load_text_vad_model(text_model_path, device)
        image_model = _load_image_vad_model(image_model_path, device)
        audio_model = _load_audio_vad_model(audio_model_path, device)

        # 3) predict each modality
        results: Dict[str, Dict[str, float]] = {}

        # text: if no subtitles, skip
        if subtitle_text.strip():
            text_preds = text_model.predict_dimensions([subtitle_text], batch_size=1)
            results["text"] = {dim: float(text_preds[dim][0]) for dim in emotion_dimensions}
        else:
            results["text"] = {}

        # image: average over sampled frames
        image_preds_list = []
        for fp in frame_paths:
            img_tensor = _preprocess_image_for_vad(fp, device)
            pred = image_model(img_tensor).detach().cpu().numpy().reshape(-1)
            if pred.shape[0] >= 3:
                image_preds_list.append(pred[:3])
        if image_preds_list:
            image_avg = np.mean(np.stack(image_preds_list, axis=0), axis=0)
            results["image"] = {"valence": float(image_avg[0]), "arousal": float(image_avg[1]), "dominance": float(image_avg[2])}
        else:
            results["image"] = {}

        # audio: chunk and average
        waveform, sr = torchaudio.load(wav_path)  # [C,T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        target_sr = 16000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        max_samples = int(target_sr * audio_chunk_seconds)
        total_len = waveform.shape[1]
        n_chunks = max(1, total_len // max_samples)
        n_chunks = min(n_chunks, 20)  # 防止太长视频慢

        audio_preds_list = []
        for i in range(n_chunks):
            start = i * max_samples
            end = min(total_len, (i + 1) * max_samples)
            chunk = waveform[:, start:end]
            if chunk.shape[1] < max_samples:
                chunk = torch.nn.functional.pad(chunk, (0, max_samples - chunk.shape[1]))
            chunk = chunk.unsqueeze(0).to(device)  # [1,1,T]
            pred = audio_model(chunk).detach().cpu().numpy().reshape(-1)
            if pred.shape[0] >= 3:
                audio_preds_list.append(pred[:3])
        if audio_preds_list:
            audio_avg = np.mean(np.stack(audio_preds_list, axis=0), axis=0)
            results["audio"] = {"valence": float(audio_avg[0]), "arousal": float(audio_avg[1]), "dominance": float(audio_avg[2])}
        else:
            results["audio"] = {}

        # 4) fusion (late fusion weighted sum)
        fusion_w = {"text": text_weight, "audio": audio_weight, "image": image_weight}
        valid = [k for k in ["text", "audio", "image"] if results.get(k) and len(results[k]) == 3]
        if not valid:
            results["fusion"] = {}
            return results

        total_w = sum(fusion_w[k] for k in valid)
        fused = np.zeros(3, dtype=np.float32)
        for k in valid:
            w = fusion_w[k] / total_w
            fused += w * np.array([results[k]["valence"], results[k]["arousal"], results[k]["dominance"]], dtype=np.float32)

        results["fusion"] = {"valence": float(fused[0]), "arousal": float(fused[1]), "dominance": float(fused[2])}
        return results
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def format_vad_result(vad: Dict[str, float]) -> str:
    if not vad:
        return "（该模态缺失）"
    return (
        f"valence: {vad['valence']:.4f}\n"
        f"arousal: {vad['arousal']:.4f}\n"
        f"dominance: {vad['dominance']:.4f}\n"
        f"解释: {interpret_vad_common(vad)}"
    )

