from __future__ import annotations

import argparse
import re
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_DIR / "Data" / "dataset_construction" / "fixedsplit_dataset.parquet"

ZH_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
EN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
NON_TEXT_PATTERN = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")
DEFAULT_ZH_STOPWORDS = {
    "对","就","将","并","等",'的', '了', '和', '是', '在', '就', '都', '而', '及', '与', '着', '或', '吗',
    '我们', '你们', '他们', '以及', '有关', '相关', '对于', '这个', '那个', '这些', '那些',
    '年', '月', '日', '时', '分', '秒', '上午', '下午', '晚上', '今天', '昨天', '明天',
    '现在', '之前', '之后', '已经', '还', '正在', '将', '要', '可以', '不能', '应该', '可能',
    '因为', '所以', '如果', '虽然', '但是', '而且', '因此', '不过', '一个', '记者',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '2025', '2026', '商务部', '商政部', '国务院', '公告', '关于', '发言人',
    '新闻', '答记者问', '国开行', '公布', '发布', '指出', '表示', '认为', '称', '介绍',
    '情况', '工作', '中国', '部长', '负责人', '召开', '国际', '发展', '通知',
    '王文涛', '习近平', '李强', '会见', '会议', '举行', '出席', '新闻发布会', '记者会',
    '特派员', '主持'
}
CUSTOM_EN_STOPWORDS = {
    # Base stopwords aligned with TF-IDF_title.ipynb
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "on", "in", "to", "of",
    "with", "by", "at", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "their", "his", "her", "we", "you", "they",
    "our", "your", "can", "could", "should", "would", "will", "may", "might", "not",
    # Additional project-specific words requested by user
    "china", "xi", "jinping", "people", "new", "th", "development",
    "council", "year", "notice", "meeting", "meets", "office", "held",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF title classifier with separate Chinese/English processing."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input parquet path. Default uses Data/dataset_construction/fixedsplit_dataset.parquet.",
    )
    parser.add_argument(
        "--zh-min-df",
        type=int,
        default=2,
        help="min_df for Chinese TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--en-min-df",
        type=int,
        default=2,
        help="min_df for English TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--top-k-rank",
        type=int,
        default=30,
        help="Top-K features to print for TF-IDF rank tables.",
    )
    return parser.parse_args()


def clean_title_text(text: object) -> str:
    raw_text = "" if pd.isna(text) else str(text)
    cleaned = NON_TEXT_PATTERN.sub(" ", raw_text)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize_zh(text: str, zh_stopwords: set[str]) -> list[str]:
    zh_only = "".join(ZH_PATTERN.findall(text))
    if not zh_only:
        return []
    tokens = [token.strip() for token in jieba.cut(zh_only, cut_all=False)]
    return [token for token in tokens if token and token not in zh_stopwords and len(token) > 1]


def tokenize_en(text: str, en_stopwords: set[str]) -> list[str]:
    tokens = [token.lower() for token in EN_PATTERN.findall(text)]
    return [token for token in tokens if token not in en_stopwords and len(token) > 1]


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def safe_prauc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, y_prob),
        "prauc": safe_prauc(y_true, y_prob),
    }


def choose_threshold_by_valid(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def print_dataset_distribution(model_df: pd.DataFrame) -> None:
    distribution_df = (
        model_df.groupby("dataset", dropna=False)
        .agg(
            rows=("label", "size"),
            label_1_count=("label", lambda values: int((values == 1).sum())),
            label_0_count=("label", lambda values: int((values == 0).sum())),
        )
        .reset_index()
        .sort_values("dataset", key=lambda col: col.map({"train": 0, "valid": 1, "test": 2}))
    )
    distribution_df["label_1_rate"] = (distribution_df["label_1_count"] / distribution_df["rows"]).round(4)
    print("\nDataset distribution:")
    print(distribution_df.to_string(index=False))


def build_rank_table(
    matrix: sparse.csr_matrix,
    feature_names: np.ndarray,
    channel: str,
    top_k: int,
) -> pd.DataFrame:
    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    non_zero_mask = mean_scores > 0
    rank_df = pd.DataFrame(
        {
            "channel": channel,
            "feature": feature_names[non_zero_mask],
            "mean_tfidf": mean_scores[non_zero_mask],
        }
    )
    rank_df["ngram"] = rank_df["feature"].map(lambda token: "unigram" if " " not in token else "bigram")
    rank_df = rank_df.sort_values("mean_tfidf", ascending=False).head(top_k).reset_index(drop=True)
    return rank_df


def fit_channel_vectorizer(
    train_series: pd.Series,
    vectorizer: TfidfVectorizer,
) -> tuple[sparse.csr_matrix, TfidfVectorizer | None]:
    normalized = train_series.fillna("").astype(str)
    if normalized.str.strip().eq("").all():
        return sparse.csr_matrix((len(normalized), 0)), None
    return vectorizer.fit_transform(normalized), vectorizer


def transform_channel(
    series: pd.Series,
    vectorizer: TfidfVectorizer | None,
) -> sparse.csr_matrix:
    normalized = series.fillna("").astype(str)
    if vectorizer is None:
        return sparse.csr_matrix((len(normalized), 0))
    return vectorizer.transform(normalized)


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    required_columns = {"dataset", "title", "label", "source_url"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    model_df = df.copy()
    model_df = model_df.loc[model_df["dataset"].isin(["train", "valid", "test"])].copy()
    model_df["label"] = pd.to_numeric(model_df["label"], errors="coerce")
    model_df = model_df.loc[model_df["label"].isin([0, 1])].copy()
    model_df["label"] = model_df["label"].astype(int)
    model_df = model_df.drop_duplicates(subset=["source_url"], keep="first").copy()
    if model_df.empty:
        raise ValueError("No usable rows after filtering dataset/label/source_url.")

    en_stopwords = set(ENGLISH_STOP_WORDS) | {word.strip().lower() for word in CUSTOM_EN_STOPWORDS}
    zh_stopwords = set(DEFAULT_ZH_STOPWORDS)
    model_df["title_cleaned"] = model_df["title"].map(clean_title_text)
    model_df["title_zh_tokens"] = model_df["title_cleaned"].map(lambda text: tokenize_zh(text, zh_stopwords))
    model_df["title_en_tokens"] = model_df["title_cleaned"].map(lambda text: tokenize_en(text, en_stopwords))
    model_df["title_zh_processed"] = model_df["title_zh_tokens"].map(lambda tokens: " ".join(tokens))
    model_df["title_en_processed"] = model_df["title_en_tokens"].map(lambda tokens: " ".join(tokens))

    train_df = model_df.loc[model_df["dataset"].eq("train")].copy()
    valid_df = model_df.loc[model_df["dataset"].eq("valid")].copy()
    test_df = model_df.loc[model_df["dataset"].eq("test")].copy()
    if train_df.empty:
        raise ValueError("Training split is empty.")
    if train_df["label"].nunique() < 2:
        raise ValueError("Train labels contain only one class; cannot fit classifier.")

    print("TF-IDF pipeline: cleaned title -> Chinese/English split -> stopword removal -> TF-IDF.")
    print(f"Input path: {input_path}")
    print(f"Rows after cleaning/dedup: {len(model_df)}")
    print_dataset_distribution(model_df)

    zh_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=None,
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=args.zh_min_df,
    )
    en_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=args.en_min_df)

    X_train_zh, zh_vectorizer_fitted = fit_channel_vectorizer(train_df["title_zh_processed"], zh_vectorizer)
    X_train_en, en_vectorizer_fitted = fit_channel_vectorizer(train_df["title_en_processed"], en_vectorizer)
    zh_rank_df = (
        build_rank_table(X_train_zh, zh_vectorizer_fitted.get_feature_names_out(), "zh", args.top_k_rank)
        if zh_vectorizer_fitted is not None
        else pd.DataFrame(columns=["channel", "feature", "mean_tfidf", "ngram"])
    )
    en_rank_df = (
        build_rank_table(X_train_en, en_vectorizer_fitted.get_feature_names_out(), "en", args.top_k_rank)
        if en_vectorizer_fitted is not None
        else pd.DataFrame(columns=["channel", "feature", "mean_tfidf", "ngram"])
    )
    print("\nChinese TF-IDF uni+bigram rank:")
    print(zh_rank_df.to_string(index=False) if not zh_rank_df.empty else "No Chinese features available.")
    print("\nEnglish TF-IDF uni+bigram rank:")
    print(en_rank_df.to_string(index=False) if not en_rank_df.empty else "No English features available.")

    def transform_and_score(split_df: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray]:
        zh_input = split_df["title_zh_processed"]
        en_input = split_df["title_en_processed"]
        X_split = sparse.hstack(
            [
                transform_channel(zh_input, zh_vectorizer_fitted),
                transform_channel(en_input, en_vectorizer_fitted),
            ],
            format="csr",
        )
        return X_split, np.array([])

    X_train, _ = transform_and_score(train_df)
    y_train = train_df["label"].to_numpy()
    X_valid, _ = transform_and_score(valid_df) if not valid_df.empty else (sparse.csr_matrix((0, X_train.shape[1])), np.array([]))
    y_valid = valid_df["label"].to_numpy() if not valid_df.empty else np.array([], dtype=int)

    candidate_rows: list[dict[str, object]] = []
    best_model: LogisticRegression | None = None
    best_row: dict[str, object] | None = None
    c_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    class_weights: list[dict[int, float] | str | None] = [None, "balanced"]
    for c_value in c_values:
        for class_weight in class_weights:
            model = LogisticRegression(
                max_iter=3000,
                C=c_value,
                class_weight=class_weight,
                random_state=42,
            )
            model.fit(X_train, y_train)
            if valid_df.empty or valid_df["label"].nunique() < 2:
                valid_prauc = float("nan")
                valid_auc = float("nan")
                valid_f1_at_05 = float("nan")
            else:
                valid_prob = model.predict_proba(X_valid)[:, 1]
                valid_prauc = safe_prauc(y_valid, valid_prob)
                valid_auc = safe_auc(y_valid, valid_prob)
                valid_f1_at_05 = float(
                    f1_score(y_valid, (valid_prob >= 0.5).astype(int), zero_division=0)
                )
            row = {
                "C": c_value,
                "class_weight": class_weight if class_weight is not None else "none",
                "valid_prauc": valid_prauc,
                "valid_auc": valid_auc,
                "valid_f1_at_0.5": valid_f1_at_05,
            }
            candidate_rows.append(row)
            if best_row is None:
                best_row = row
                best_model = model
            else:
                current_key = (
                    best_row["valid_prauc"] if pd.notna(best_row["valid_prauc"]) else -1.0,
                    best_row["valid_auc"] if pd.notna(best_row["valid_auc"]) else -1.0,
                    best_row["valid_f1_at_0.5"] if pd.notna(best_row["valid_f1_at_0.5"]) else -1.0,
                )
                candidate_key = (
                    row["valid_prauc"] if pd.notna(row["valid_prauc"]) else -1.0,
                    row["valid_auc"] if pd.notna(row["valid_auc"]) else -1.0,
                    row["valid_f1_at_0.5"] if pd.notna(row["valid_f1_at_0.5"]) else -1.0,
                )
                if candidate_key > current_key:
                    best_row = row
                    best_model = model

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["valid_prauc", "valid_auc", "valid_f1_at_0.5"],
        ascending=[False, False, False],
    )
    print("\nValidation hyperparameter ranking (sorted by PR-AUC):")
    print(candidate_df.to_string(index=False))
    if best_model is None or best_row is None:
        raise RuntimeError("No valid model candidate found.")
    print(
        "\nSelected hyperparameter:",
        {
            "C": best_row["C"],
            "class_weight": best_row["class_weight"],
        },
    )

    threshold = 0.5
    if not valid_df.empty and valid_df["label"].nunique() >= 2:
        valid_prob_for_threshold = best_model.predict_proba(X_valid)[:, 1]
        threshold = choose_threshold_by_valid(y_valid, valid_prob_for_threshold)
    print(f"Selected threshold on valid by F1: {threshold:.2f}")

    metrics_rows: list[dict[str, object]] = []
    for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if split_df.empty:
            continue
        X_split, _ = transform_and_score(split_df)
        y_prob = best_model.predict_proba(X_split)[:, 1]
        y_true = split_df["label"].to_numpy()
        metric_row = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        metric_row.update(
            {
                "dataset": split_name,
                "rows": int(len(split_df)),
                "label_1_count": int((split_df["label"] == 1).sum()),
                "label_0_count": int((split_df["label"] == 0).sum()),
                "threshold": float(threshold),
            }
        )
        metrics_rows.append(metric_row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values("dataset", key=lambda col: col.map({"train": 0, "valid": 1, "test": 2}))

    print("\nFinal metrics table (train / valid / test):")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
