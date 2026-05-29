import pickle
from pathlib import Path
import numpy as np


def inspect_emotion(p: Path):
    with p.open("rb") as f:
        obj = pickle.load(f)
    y = obj["train"]["labels"]  # [N,6]
    print("emotion labels shape", y.shape, "dtype", y.dtype)
    for i in range(y.shape[1]):
        col = y[:, i]
        uniq = np.unique(col)
        print(
            f" dim{i}: unique_count={len(uniq)}, min={col.min()}, max={col.max()}, uniq_sample={uniq[:10]}"
        )


def inspect_senti(p: Path):
    with p.open("rb") as f:
        obj = pickle.load(f)
    y = obj["train"]["labels"].reshape(-1)  # [N]
    print("senti labels shape", obj["train"]["labels"].shape, "dtype", obj["train"]["labels"].dtype)
    print(" senti: min", float(y.min()), "max", float(y.max()), "mean", float(y.mean()))
    uniq = np.unique(y)
    print(" senti: unique_count", len(uniq), "uniq_sample", uniq[:10])


def main():
    root = Path(r"C:\Users\LCX\Natural Language\data\cmumosei")
    p_emotion = root / "mosei_emotion_aligned_60.pkl"
    p_senti = root / "mosei_senti_data.pkl"

    print("=== emotion ===")
    inspect_emotion(p_emotion)

    print("\n=== senti ===")
    inspect_senti(p_senti)


if __name__ == "__main__":
    main()

