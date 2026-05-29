import os
import re
import glob
import unicodedata
from pathlib import Path

import pandas as pd

# 如果未安装，先：pip install kagglehub nltk
import kagglehub

# NLTK：停用词与词干化（支持葡语/英语）
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, PorterStemmer


def ensure_nltk_resources():
    try:
        stopwords.words("portuguese")
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def strip_accents(text: str) -> str:
    # 去除重音/变音符号（é -> e）
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<.*?>")
# 语言特定的非字母过滤
NON_ALPHA_PATTERN_PT = re.compile(r"[^a-záàâãéêíóôõúç ]", re.IGNORECASE)
NON_ALPHA_PATTERN_EN = re.compile(r"[^a-z ]", re.IGNORECASE)
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def build_text_cleaner(remove_accents: bool = True, use_stem: bool = False, language: str = "pt"):
    ensure_nltk_resources()
    lang = language.lower()
    if lang not in {"pt", "en"}:
        raise ValueError("language 仅支持 'pt' 或 'en'")
    stop_set = set(stopwords.words("english" if lang == "en" else "portuguese"))
    if use_stem:
        stemmer = PorterStemmer() if lang == "en" else RSLPStemmer()
    else:
        stemmer = None

    def clean_text_pt(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = URL_PATTERN.sub(" ", text)
        text = HTML_TAG_PATTERN.sub(" ", text)
        # 可选先去重音，随后按语言过滤
        if remove_accents:
            text = strip_accents(text)
        # 根据语言保留字母和空格
        if lang == "en":
            text = NON_ALPHA_PATTERN_EN.sub(" ", text)
        else:
            text = NON_ALPHA_PATTERN_PT.sub(" ", text)
        text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
        # 分词、去停用词、可选词干化
        tokens = [tok for tok in text.split() if tok and tok not in stop_set]
        if stemmer is not None:
            tokens = [stemmer.stem(tok) for tok in tokens]
        return " ".join(tokens)

    return clean_text_pt


def find_first_csv(path_dir: str) -> str:
    # 在下载目录下递归寻找第一个 CSV
    candidates = glob.glob(os.path.join(path_dir, "**", "*.csv"), recursive=True)
    if not candidates:
        raise FileNotFoundError("未在数据集目录中找到任何 CSV 文件。")
    # 优先选择较大的文件（更可能是主数据）
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


def load_dataset(csv_path: str, text_column: str = "review", label_column: str | None = "sentiment") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    # 简单兜底：如果默认列不存在，提示可用列名
    cols = set(df.columns.str.lower())
    if text_column not in cols:
        print(f"提示：当前列名为 {list(df.columns)}，未发现 '{text_column}'，请手动指定 text_column。")
    # 统一小写列名，便于访问
    df.columns = [c.lower() for c in df.columns]
    # 去除空文本
    if text_column in df.columns:
        df = df[~df[text_column].isna()].copy()
    return df


def preprocess_and_save(
    df: pd.DataFrame,
    text_column: str = "review",
    label_column: str | None = "sentiment",
    output_path: str = "imdb_pt_clean.csv",
    remove_accents: bool = True,
    use_stem: bool = False,
    language: str = "pt",
) -> str:
    cleaner = build_text_cleaner(remove_accents=remove_accents, use_stem=use_stem, language=language)

    if text_column not in df.columns:
        raise KeyError(f"未找到文本列 '{text_column}'，可用列：{list(df.columns)}")

    df["text_clean"] = df[text_column].astype(str).map(cleaner)
    # 去除清洗后为空的样本
    df = df[df["text_clean"].str.len() > 0].copy()

    # 只保留需要的列（如存在标签列）
    keep_cols = ["text_clean"]
    if label_column and label_column in df.columns:
        keep_cols.append(label_column)
    df_out = df[keep_cols].reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    # 1) 下载数据集
    path = kagglehub.dataset_download("anairamcosta/imdb-reviews-pt-br-csv")
    print("Path to dataset files:", path)

    # 2) 定位 CSV 并读取
    csv_file = find_first_csv(path)
    print("Detected CSV:", csv_file)

    # 3) 加载并预处理（按需修改列名）
    # 若要处理英文，请使用 text_en；葡语使用 text_pt
    df_raw = load_dataset(csv_file, text_column="text_en", label_column="sentiment")

    # 4) 预处理并保存
    out_file = preprocess_and_save(
        df_raw,
        text_column="text_en",       # 英文文本列
        label_column="sentiment",    # 若无标签列，设为 None
        output_path="data/imdb_en_clean.csv",
        remove_accents=True,
        use_stem=False,               # 若想更强归一化，可设为 True
        language="en",
    )
    print("Cleaned file saved to:", out_file)


