from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
DEFAULT_LOCAL_MODEL_DIR = PROJECT_DIR / "models" / "bge-m3"
DEFAULT_REMOTE_MODEL = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a title classifier using bge-m3 embeddings on fixed split dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input parquet path. Default uses Data/dataset_construction/fixedsplit_dataset.parquet.",
    )
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        default=DEFAULT_LOCAL_MODEL_DIR,
        help="Local bge-m3 directory. If unavailable, script falls back to remote BAAI/bge-m3.",
    )
    parser.add_argument(
        "--remote-model-name",
        type=str,
        default=DEFAULT_REMOTE_MODEL,
        help="Remote HuggingFace model id for fallback.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Embedding device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length passed to embedding model.",
    )
    return parser.parse_args()


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


def load_and_prepare_df(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    required_columns = {"dataset", "title", "label", "source_url"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    model_df = df.copy()
    model_df = model_df.loc[model_df["dataset"].isin(["train", "valid", "test"])].copy()
    model_df["title"] = model_df["title"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    model_df["label"] = pd.to_numeric(model_df["label"], errors="coerce")
    model_df = model_df.loc[model_df["label"].isin([0, 1])].copy()
    model_df["label"] = model_df["label"].astype(int)
    model_df["source_url"] = model_df["source_url"].fillna("").astype(str).str.strip()
    model_df = model_df.loc[model_df["title"].ne("")].copy()
    model_df = model_df.drop_duplicates(subset=["source_url"], keep="first").copy()

    if model_df.empty:
        raise ValueError("No usable rows after filtering dataset/label/title/source_url.")
    return model_df


def split_by_dataset(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = model_df.loc[model_df["dataset"].eq("train")].copy()
    valid_df = model_df.loc[model_df["dataset"].eq("valid")].copy()
    test_df = model_df.loc[model_df["dataset"].eq("test")].copy()
    if train_df.empty:
        raise ValueError("Training split is empty.")
    if train_df["label"].nunique() < 2:
        raise ValueError("Train labels contain only one class; cannot fit classifier.")
    return train_df, valid_df, test_df


def load_sentence_transformer(local_model_dir: Path, remote_model_name: str, device: str) -> tuple[object, str, str]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'sentence-transformers'. Please install it first, e.g. "
            "'pip install sentence-transformers'."
        ) from exc

    requested_device = (device or "cuda").strip().lower() or "cuda"
    resolved_local = local_model_dir.resolve()

    def _load_on_device(target_device: str) -> tuple[object, str]:
        if resolved_local.exists():
            print(f"Loading embedding model from local path: {resolved_local}")
            model = SentenceTransformer(str(resolved_local), device=target_device, local_files_only=True)
            return model, "local"
        print(
            f"Local model not found at {resolved_local}. "
            f"Falling back to remote model: {remote_model_name}"
        )
        model = SentenceTransformer(remote_model_name, device=target_device)
        return model, "remote"

    try:
        model, source = _load_on_device(requested_device)
        return model, source, requested_device
    except Exception as exc:
        error_text = str(exc).lower()
        should_fallback_to_cpu = requested_device.startswith("cuda") and (
            "cuda" in error_text
            or "cudnn" in error_text
            or "not compiled with cuda" in error_text
        )
        if not should_fallback_to_cpu:
            raise RuntimeError(f"Failed to load embedding model: {exc}") from exc
        print(f"Failed to initialize model on '{requested_device}', fallback to CPU. Error: {exc}")
        model, source = _load_on_device("cpu")
        return model, source, "cpu"


def encode_titles(
    model: object,
    titles: list[str],
    batch_size: int,
) -> np.ndarray:
    if not titles:
        return np.zeros((0, 0), dtype=np.float32)
    return model.encode(
        titles,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()

    model_df = load_and_prepare_df(input_path)
    train_df, valid_df, test_df = split_by_dataset(model_df)
    print("BGE-M3 title classification pipeline: title -> embedding -> logistic regression.")
    print(f"Input path: {input_path}")
    print(f"Rows after cleaning/dedup: {len(model_df)}")
    print_dataset_distribution(model_df)

    model, model_source, device_in_use = load_sentence_transformer(
        local_model_dir=args.local_model_dir,
        remote_model_name=args.remote_model_name,
        device=args.device,
    )
    if hasattr(model, "max_seq_length"):
        try:
            model.max_seq_length = int(args.max_length)
            print(f"Embedding max sequence length set to: {model.max_seq_length}")
        except Exception as exc:
            print(f"Warning: failed to set model.max_seq_length={args.max_length}: {exc}")
    print(f"\nEmbedding model source: {model_source}")
    print(f"Embedding device in use: {device_in_use}")

    X_train = encode_titles(
        model=model,
        titles=train_df["title"].tolist(),
        batch_size=args.batch_size,
    )
    y_train = train_df["label"].to_numpy()
    X_valid = encode_titles(
        model=model,
        titles=valid_df["title"].tolist(),
        batch_size=args.batch_size,
    )
    y_valid = valid_df["label"].to_numpy() if not valid_df.empty else np.array([], dtype=int)
    X_test = encode_titles(
        model=model,
        titles=test_df["title"].tolist(),
        batch_size=args.batch_size,
    )

    candidate_rows: list[dict[str, object]] = []
    best_model: LogisticRegression | None = None
    best_row: dict[str, object] | None = None
    c_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    class_weights: list[str | None] = [None, "balanced"]

    for c_value in c_values:
        for class_weight in class_weights:
            clf = LogisticRegression(
                max_iter=3000,
                C=c_value,
                class_weight=class_weight,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            if valid_df.empty or valid_df["label"].nunique() < 2:
                valid_prauc = float("nan")
                valid_auc = float("nan")
                valid_f1_at_05 = float("nan")
            else:
                valid_prob = clf.predict_proba(X_valid)[:, 1]
                valid_prauc = safe_prauc(y_valid, valid_prob)
                valid_auc = safe_auc(y_valid, valid_prob)
                valid_f1_at_05 = float(
                    f1_score(y_valid, (valid_prob >= 0.5).astype(int), zero_division=0)
                )

            row = {
                "C": c_value,
                "class_weight": "none" if class_weight is None else class_weight,
                "valid_prauc": valid_prauc,
                "valid_auc": valid_auc,
                "valid_f1_at_0.5": valid_f1_at_05,
            }
            candidate_rows.append(row)

            if best_row is None:
                best_row = row
                best_model = clf
            else:
                best_key = (
                    best_row["valid_prauc"] if pd.notna(best_row["valid_prauc"]) else -1.0,
                    best_row["valid_auc"] if pd.notna(best_row["valid_auc"]) else -1.0,
                    best_row["valid_f1_at_0.5"] if pd.notna(best_row["valid_f1_at_0.5"]) else -1.0,
                )
                curr_key = (
                    row["valid_prauc"] if pd.notna(row["valid_prauc"]) else -1.0,
                    row["valid_auc"] if pd.notna(row["valid_auc"]) else -1.0,
                    row["valid_f1_at_0.5"] if pd.notna(row["valid_f1_at_0.5"]) else -1.0,
                )
                if curr_key > best_key:
                    best_row = row
                    best_model = clf

    if best_model is None or best_row is None:
        raise RuntimeError("No valid model candidate found.")

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["valid_prauc", "valid_auc", "valid_f1_at_0.5"],
        ascending=[False, False, False],
    )
    print("\nValidation hyperparameter ranking (sorted by PR-AUC):")
    print(candidate_df.to_string(index=False))
    print(
        "\nSelected hyperparameter:",
        {"C": best_row["C"], "class_weight": best_row["class_weight"]},
    )

    threshold = 0.5
    if not valid_df.empty and valid_df["label"].nunique() >= 2:
        valid_prob_for_threshold = best_model.predict_proba(X_valid)[:, 1]
        threshold = choose_threshold_by_valid(y_valid, valid_prob_for_threshold)
    print(f"Selected threshold on valid by F1: {threshold:.2f}")

    metrics_rows: list[dict[str, object]] = []
    split_payload = [
        ("train", train_df, X_train),
        ("valid", valid_df, X_valid),
        ("test", test_df, X_test),
    ]
    for split_name, split_df, X_split in split_payload:
        if split_df.empty:
            continue
        y_true = split_df["label"].to_numpy()
        y_prob = best_model.predict_proba(X_split)[:, 1]
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

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        "dataset",
        key=lambda col: col.map({"train": 0, "valid": 1, "test": 2}),
    )
    print("\nFinal metrics table (train / valid / test):")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
