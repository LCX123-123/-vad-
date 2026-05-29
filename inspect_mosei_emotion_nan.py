import pickle
from pathlib import Path
import numpy as np


def stats(name, arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, nan={int(np.isnan(arr).sum())}, inf={int(np.isinf(arr).sum())}"
    )


def main():
    p = Path(r"C:\Users\LCX\Natural Language\data\cmumosei\mosei_emotion_aligned_60.pkl")
    with p.open("rb") as f:
        obj = pickle.load(f)
    tr = obj["train"]
    print("=== MOSEI emotion NaN/Inf check ===")
    stats("vision", tr["vision"])
    stats("audio", tr["audio"])
    stats("text", tr["text"])
    stats("labels", tr["labels"])


if __name__ == "__main__":
    main()

