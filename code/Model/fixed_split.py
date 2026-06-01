from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_DIR / "Data" / "dataset_construction" / "null_signal_dataset.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Data" / "dataset_construction"
DEFAULT_OUTPUT_FILENAME = "fixedsplit_dataset.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split null_signal_dataset into time-based train/valid/test datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input parquet path (default: Data/dataset_construction/null_signal_dataset.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for datasplit parquet.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output parquet filename.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, valid_ratio, test_ratio]
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be > 0.")
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")


def assign_time_split(df: pd.DataFrame, train_ratio: float, valid_ratio: float) -> pd.DataFrame:
    if "published_date" not in df.columns:
        raise KeyError("Missing required column: published_date")
    if "source_url" not in df.columns:
        raise KeyError("Missing required column: source_url")

    split_df = df.copy()
    split_df["published_date"] = pd.to_datetime(split_df["published_date"], errors="coerce")
    split_df = split_df.loc[split_df["published_date"].notna()].copy()
    split_df = split_df.sort_values(["published_date", "source_url"]).reset_index(drop=True)

    total_rows = len(split_df)
    if total_rows == 0:
        split_df["dataset"] = pd.Series(dtype="object")
        return split_df

    train_cutoff = int(total_rows * train_ratio)
    valid_cutoff = int(total_rows * (train_ratio + valid_ratio))
    train_cutoff = max(train_cutoff, 1)
    valid_cutoff = min(max(valid_cutoff, train_cutoff + 1), total_rows)

    split_df["dataset"] = "test"
    split_df.loc[: train_cutoff - 1, "dataset"] = "train"
    split_df.loc[train_cutoff: valid_cutoff - 1, "dataset"] = "valid"
    return split_df


def print_split_summary(split_df: pd.DataFrame) -> None:
    print("Default split strategy: time-based split into train/valid/test.")
    if split_df.empty:
        print("No rows available after parsing published_date.")
        return

    date_ranges = (
        split_df.groupby("dataset", dropna=False)
        .agg(
            doc_amount=("source_url", "size"),
            min=("published_date", "min"),
            max=("published_date", "max"),
        )
        .reset_index()
        .sort_values("dataset", key=lambda col: col.map({"train": 0, "valid": 1, "test": 2}))
    )
    print("\nDate range by dataset:")
    print(date_ranges.to_string(index=False))


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    df = pd.read_parquet(input_path)
    split_df = assign_time_split(df, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)

    # Keep one row per source_url in final output.
    split_df = split_df.drop_duplicates(subset=["source_url"], keep="first").copy()

    print_split_summary(split_df)
    split_df.to_parquet(output_path, index=False)

    print(f"\nInput path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Rows written: {len(split_df)}")


if __name__ == "__main__":
    main()
