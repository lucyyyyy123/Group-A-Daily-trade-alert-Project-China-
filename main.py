from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Data"
CODE_DIR = ROOT_DIR / "code"

GTA_DOWNLOADED_XLSX = DATA_DIR / "China_GTA_Downloaded" / "interventions.xlsx"
BILBY_COMBINED_PARQUET = DATA_DIR / "Bilby_data_fixed" / "Bilby_2025_2026_combined.parquet"

GTA_SOURCE_SCRAPE_SCRIPT = CODE_DIR / "data_preprocessing" / "GTA_Source_Scrape.py"
BILBY_COMBINE_SCRIPT = CODE_DIR / "data_preprocessing" / "Bilby_data_combine.py"
GTA_BILBY_CLEAN_SCRIPT = CODE_DIR / "data_preprocessing" / "GTA_Bilby_clean.py"
EXACT_MATCH_SCRIPT = CODE_DIR / "data_match" / "exact_match.py"
EMBEDDING_MATCH_SCRIPT = CODE_DIR / "data_match" / "embedding_match.py"
NULL_SIGNAL_SCRIPT = CODE_DIR / "Model" / "null_signal_dataset.py"
FIXED_SPLIT_SCRIPT = CODE_DIR / "Model" / "fixed_split.py"
TFIDF_LOGREG_SCRIPT = CODE_DIR / "Model" / "TF-IDF_title+logistic_regression.py"
BERT_SCRIPT = CODE_DIR / "Model" / "BERT_title.py"


def print_stage(title: str) -> None:
    # Print a clear visual separator for each pipeline stage.
    print(f"\n{'=' * 88}")
    print(title)
    print(f"{'=' * 88}")


def ask_yes_no(question: str, default: bool = False) -> bool:
    # Keep the prompt lowercase for consistency.
    prompt = " [y/n]: "

    while True:
        # Read and normalize user input once for robust matching.
        answer = input(f"{question}{prompt}").strip().lower()
        if not answer:
            return default
        # Accept both English and Chinese answers for convenience.
        if answer in {"y", "yes", "是", "好", "需要"}:
            return True
        if answer in {"n", "no", "否", "不", "不需要"}:
            return False
        print("Please enter y/n (or yes/no).")


def run_script(script_path: Path) -> None:
    # Resolve and validate script path before execution.
    resolved_path = script_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Script not found: {resolved_path}")

    # Execute child script using the current Python interpreter.
    print(f"\n>> Running: {resolved_path}")
    result = subprocess.run([sys.executable, str(resolved_path)], cwd=ROOT_DIR)

    # Stop the pipeline immediately if any step fails.
    if result.returncode != 0:
        raise RuntimeError(f"Script failed ({result.returncode}): {resolved_path}")
    print(">> DONE")


def summarize_dataframe_date_range(df: pd.DataFrame) -> tuple[int, str]:
    # Keep record count regardless of whether a date column can be parsed.
    row_count = len(df)
    if df.empty:
        return row_count, "File exists, but it contains no records."

    # Prefer published_date, then fall back to date-like column names.
    preferred_columns = [column for column in df.columns if str(column).strip().lower() == "published_date"]
    candidate_columns = []
    candidate_columns.extend(preferred_columns)
    for column in df.columns:
        name = str(column).strip().lower()
        if any(keyword in name for keyword in ["date", "time", "published", "日期", "时间"]):
            if column not in candidate_columns:
                candidate_columns.append(column)

    if not candidate_columns:
        return row_count, f"Total records: {row_count}; no date-like column was detected."

    # Select the date column with the most parseable values.
    best_column = None
    best_series = None
    best_non_null = -1
    for column in candidate_columns:
        series = pd.to_datetime(df[column], errors="coerce")
        non_null_count = int(series.notna().sum())
        if non_null_count > best_non_null:
            best_non_null = non_null_count
            best_column = column
            best_series = series

    if best_series is None or best_non_null <= 0:
        return row_count, f"Total records: {row_count}; date column exists but values are not parseable."

    # Return min/max date coverage for user confirmation before re-scraping.
    min_date = best_series.min()
    max_date = best_series.max()
    return row_count, (
        f"Total records: {row_count}; time range ({best_column}): "
        f"{min_date.date()} ~ {max_date.date()}."
    )


def summarize_excel_date_range(file_path: Path) -> tuple[int | None, str]:
    # Handle missing source file gracefully so the workflow can continue.
    if not file_path.exists():
        return None, f"File not found: {file_path}"

    # Read the Excel file and return an informative error if reading fails.
    try:
        df = pd.read_excel(file_path)
    except Exception as exc:
        return None, f"Failed to read file: {file_path} ({type(exc).__name__}: {exc})"

    return summarize_dataframe_date_range(df)


def summarize_parquet_date_range(file_path: Path) -> tuple[int | None, str]:
    # Handle missing parquet file gracefully so the workflow can continue.
    if not file_path.exists():
        return None, f"File not found: {file_path}"

    # Read the parquet file and return an informative error if reading fails.
    try:
        df = pd.read_parquet(file_path)
    except Exception as exc:
        return None, f"Failed to read file: {file_path} ({type(exc).__name__}: {exc})"

    return summarize_dataframe_date_range(df)

def main() -> None:
    # Track optional branch choices for final summary output.
    optional_steps: dict[str, bool] = {
        "rescrape": False,
        "recombine": False,
        "embedding_match": False,
    }

    # Stage 0: Print a pipeline header and the resolved project root path.
    # This gives the user immediate context about where scripts and data are read from.
    print_stage("Bilby Pipeline One-Click Controller")
    print(f"Project root: {ROOT_DIR}")

    # Step 1/7: Check the GTA interventions source file status.
    # We report record count and date coverage first, then ask whether to refresh GTA Source by scraping again.
    print_stage("Step 1/7 - Check GTA downloaded file and decide rescrape")
    row_count, message = summarize_excel_date_range(GTA_DOWNLOADED_XLSX)
    print(f"GTA downloaded file: {GTA_DOWNLOADED_XLSX}")
    print(message)
    if row_count is None:
        print("Reminder: please download interventions.xlsx from GTA website into Data/China_GTA_Downloaded.")

    # Optional branch A: re-scrape GTA source pages and rebuild source outputs.
    # Skip this when the existing source outputs are already up to date.
    if ask_yes_no("Do you want to re-scrape GTA Source?", default=False):
        optional_steps["rescrape"] = True
        print(
            "You chose rescrape. Please make sure interventions.xlsx is updated from the official website "
            "and placed under Data/China_GTA_Downloaded."
        )
        run_script(GTA_SOURCE_SCRAPE_SCRIPT)
    else:
        print("Skipping GTA rescrape.")

    # Step 2/7: Decide whether to recombine Bilby 2025 + 2026 data.
    # Recombining is optional and mainly needed when source Bilby files changed.
    print_stage("Step 2/7 - Bilby 2025+2026 combine")
    print(
        "By default, Bilby 2025 and 2026 data are combined into:\n"
        f"{BILBY_COMBINED_PARQUET}"
    )
    bilby_row_count, bilby_message = summarize_parquet_date_range(BILBY_COMBINED_PARQUET)
    print(bilby_message)
    if bilby_row_count is None:
        print("Reminder: run Bilby combine to generate the combined parquet file before downstream steps.")

    # Optional branch B: regenerate the combined Bilby parquet file.
    if ask_yes_no("Do you want to recombine Bilby 2025+2026 data?", default=False):
        optional_steps["recombine"] = True
        run_script(BILBY_COMBINE_SCRIPT)
    else:
        print("Skipping Bilby recombine.")

    # Step 3/7: Clean and normalize GTA and Bilby datasets.
    # This prepares consistent fields so downstream matching can be more reliable.
    print_stage("Step 3/7 - Clean GTA and Bilby data")
    print("Next step: clean both datasets.")
    run_script(GTA_BILBY_CLEAN_SCRIPT)

    # Step 4/7: Run baseline exact matching using title + URL rules.
    # This creates the initial high-precision match set before optional semantic matching.
    print_stage("Step 4/7 - Data Match (baseline: title+url exact match)")
    print("Running baseline exact match now.")
    run_script(EXACT_MATCH_SCRIPT)

    # Step 5/7: Optionally run embedding-based matching for higher recall.
    # This is slower but can recover matches missed by exact string rules.
    print_stage("Step 5/7 - Optional embedding match")
    print(
        "Embedding match can significantly improve matched rate. "
        "First run is usually slow (about 1.5 hours for current data size)."
    )
    print(
        "Later runs are typically faster if embedding cache is reused and the data range is unchanged."
    )
    print(
        "Current embedding info in matched output corresponds to 2025-06-01 ~ 2026-03-01; "
        "if your date range changed, rerunning embedding is recommended."
    )

    # Optional branch C: execute embedding matching and append richer match signals.
    if ask_yes_no("Do you want to run embedding match now? (first run may take ~1.5h)", default=False):
        optional_steps["embedding_match"] = True
        run_script(EMBEDDING_MATCH_SCRIPT)
    else:
        print("Skipping embedding match.")

    # Step 6/7: Generate the labeled dataset used by modeling scripts.
    # This converts matched/cleaned data into a training-ready labeled table.
    print_stage("Step 6/7 - Label dataset")
    print("Next step: add labels to the data.")
    run_script(NULL_SIGNAL_SCRIPT)

    # Step 7/7: Split data and train both baseline and BERT models.
    # We keep fixed split for comparability, then run TF-IDF+LogReg and BERT pipelines sequentially.
    print_stage("Step 7/7 - Datasplit + model training")
    print("Using fixed datasplit by default.")
    run_script(FIXED_SPLIT_SCRIPT)

    # Train the classical TF-IDF + Logistic Regression baseline.
    print("Training TF-IDF + Logistic Regression.")
    run_script(TFIDF_LOGREG_SCRIPT)

    # Train the BERT title-based model pipeline.
    print("Training BERT title model pipeline.")
    run_script(BERT_SCRIPT)

    # Final summary: report optional branch decisions and confirm completion.
    print_stage("Pipeline completed")
    print("Optional step execution summary:")
    print(f"- GTA rescrape: {'Executed' if optional_steps['rescrape'] else 'Skipped'}")
    print(f"- Bilby recombine: {'Executed' if optional_steps['recombine'] else 'Skipped'}")
    print(f"- Embedding match: {'Executed' if optional_steps['embedding_match'] else 'Skipped'}")
    print("All main pipeline steps are complete.")


if __name__ == "__main__":
    main()
