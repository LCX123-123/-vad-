import pickle
from pathlib import Path
import numpy as np


def summarize(obj, max_list_len=5):
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"type=dict, nkeys={len(keys)}, keys_sample={keys[:max_list_len]}")
        # 不深入展开，避免触发巨大数组
        for k in keys[:max_list_len]:
            v = obj[k]
            print(f"  key={k!r}, value_type={type(v)}")
        return
    if isinstance(obj, (list, tuple)):
        print(f"type={type(obj).__name__}, len={len(obj)}")
        if len(obj) > 0:
            print(f"  first_type={type(obj[0])}, first_repr_prefix={repr(obj[0])[:200]}")
        return
    print(f"type={type(obj)}, repr_prefix={repr(obj)[:300]}")


def main():
    root = Path(r"C:\Users\LCX\Natural Language\data\cmumosei")
    files = [
        root / "mosei_emotion_aligned_60.pkl",
        root / "mosei_senti_data.pkl",
    ]

    def summarize_deep(dict_obj):
        if not isinstance(dict_obj, dict):
            return
        keys = list(dict_obj.keys())
        print(f"inner keys={keys[:30]} (n={len(keys)})")
        for k in keys[:15]:
            v = dict_obj[k]
            if isinstance(v, np.ndarray):
                print(f"  {k}: np.ndarray shape={v.shape}, dtype={v.dtype}")
            else:
                # avoid iterating huge lists
                try:
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        print(f"  {k}: {type(v).__name__}, len={len(v)}, first_type={type(v[0])}")
                    else:
                        print(f"  {k}: {type(v)}")
                except Exception:
                    print(f"  {k}: {type(v)}")

    for p in files:
        print("\n=== Loading:", p.name, "sizeMB", p.stat().st_size / 1024 / 1024)
        try:
            with p.open("rb") as f:
                obj = pickle.load(f)
            summarize(obj)
            if isinstance(obj, dict) and "train" in obj:
                print("---- inspecting obj['train'] ----")
                summarize_deep(obj["train"])
        except MemoryError as e:
            print("MemoryError while loading:", p.name, e)
        except Exception as e:
            print("Failed loading:", p.name, "error:", e)


if __name__ == "__main__":
    main()

