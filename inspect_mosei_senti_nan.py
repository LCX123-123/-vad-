import pickle
from pathlib import Path
import numpy as np


def main():
    p = Path(r"C:\Users\LCX\Natural Language\data\cmumosei\mosei_senti_data.pkl")
    with p.open("rb") as f:
        obj = pickle.load(f)

    train = obj["train"]
    vision = train["vision"]  # [N,50,35]
    audio = train["audio"]  # [N,50,74]
    text = train["text"]  # [N,50,300]
    labels = train["labels"]  # [N,1,1]

    def stats(name, arr):
        arr = np.asarray(arr)
        finite = np.isfinite(arr)
        n_total = arr.size
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"  total={n_total}, nan={int(n_nan)}, inf={int(n_inf)}, nonfinite={int((~finite).sum())}")
        if n_total > 0:
            # 用有限值计算 min/max
            if np.any(finite):
                finite_vals = arr[finite]
                print(f"  finite min={float(finite_vals.min())}, max={float(finite_vals.max())}")
            else:
                print("  no finite values")

    print("=== MOSEI senti data NaN/Inf check ===")
    stats("vision", vision)
    stats("audio", audio)
    stats("text", text)
    stats("labels", labels)


if __name__ == "__main__":
    main()

