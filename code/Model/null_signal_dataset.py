from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MATCH_OUTPUT_DIR = PROJECT_DIR / "Data" / "match_outputs"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Data" / "dataset_construction"
EXCEL_ILLEGAL_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
DEFAULT_OUTPUT_FILENAME = "null_signal_dataset.parquet"
GTA_NOT_IN_TIME_RANGE_FILENAME = "gta_not_in_time_range_gta_not_in_time_range.parquet"
COMBINE_MATCHED_FILENAME = "combined_match_matched.parquet"
COMBINE_BILBY_UNMATCHED_FILENAME = "combined_match_bilby_unmatched.parquet"
COMBINE_GTA_UNMATCHED_FILENAME = "combined_match_gta_unmatched.parquet"
EMBEDDING_MATCHED_FILENAME = "embedding_match_matched_records_matched.parquet"
EMBEDDING_BILBY_UNMATCHED_FILENAME = "embedding_match_matched_records_bilby_unmatched.parquet"
EMBEDDING_GTA_UNMATCHED_FILENAME = "embedding_match_matched_records_gta_unmatched.parquet"

BASE_OUTPUT_COLUMNS = [
	"title",
	"source_url",
	"site_root_url",
	"summary",
	"published_date",
	"type",
]

STANDARD_OUTPUT_COLUMNS = [
	*BASE_OUTPUT_COLUMNS,
	"label",
]

def clean_root_value(value: object) -> str:
	if pd.isna(value):
		return ""
	return str(value).strip()


def sanitize_excel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	sanitized = df.copy()
	for column in sanitized.select_dtypes(include=["object", "string"]).columns:
		sanitized[column] = sanitized[column].map(
			lambda value: EXCEL_ILLEGAL_CHAR_RE.sub("", value) if isinstance(value, str) else value
		)
	return sanitized


def require_parquet_file(path: Path, label: str) -> Path:
	resolved_path = path.resolve()
	if not resolved_path.exists():
		raise FileNotFoundError(f"{label} file not found: {resolved_path}")
	return resolved_path


def load_combine_parquets(match_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
	resolved_match_dir = match_dir.resolve()
	matched_path = require_parquet_file(
		resolved_match_dir / COMBINE_MATCHED_FILENAME,
		"Combine matched",
	)
	bilby_unmatched_path = require_parquet_file(
		resolved_match_dir / COMBINE_BILBY_UNMATCHED_FILENAME,
		"Combine Bilby unmatched",
	)
	matched_df = pd.read_parquet(matched_path)
	bilby_unmatched_df = pd.read_parquet(bilby_unmatched_path)
	return matched_df, bilby_unmatched_df


def load_embedding_parquets(match_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
	resolved_match_dir = match_dir.resolve()
	matched_path = require_parquet_file(
		resolved_match_dir / EMBEDDING_MATCHED_FILENAME,
		"Embedding matched",
	)
	bilby_unmatched_path = require_parquet_file(
		resolved_match_dir / EMBEDDING_BILBY_UNMATCHED_FILENAME,
		"Embedding Bilby unmatched",
	)
	matched_df = pd.read_parquet(matched_path)
	bilby_unmatched_df = pd.read_parquet(bilby_unmatched_path)
	return matched_df, bilby_unmatched_df


def load_selected_gta_unmatched_parquet(match_dir: Path, use_embedding_source: bool) -> pd.DataFrame:
	filename = (
		EMBEDDING_GTA_UNMATCHED_FILENAME if use_embedding_source else COMBINE_GTA_UNMATCHED_FILENAME
	)
	label = "Embedding GTA unmatched" if use_embedding_source else "Combine GTA unmatched"
	path = require_parquet_file(match_dir.resolve() / filename, label)
	return pd.read_parquet(path)


def coalesce_columns(df: pd.DataFrame, candidates: list[str], fallback: object = pd.NA) -> pd.Series:
	series = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
	for column in candidates:
		if column in df.columns:
			series = series.combine_first(df[column])
	if pd.isna(fallback):
		return series
	return series.fillna(fallback)


def merge_non_empty_text(primary: object, secondary: object) -> str:
	primary_text = clean_root_value(primary)
	secondary_text = clean_root_value(secondary)
	if primary_text and secondary_text:
		return f"{primary_text} {secondary_text}"
	return primary_text or secondary_text


def prepare_bilby_output_columns(df: pd.DataFrame) -> pd.DataFrame:
	prepared_df = df.copy()
	bilby_title_series = coalesce_columns(prepared_df, ["bilby_title", "bilby_match_title"])
	bilby_title_en_series = coalesce_columns(prepared_df, ["bilby_title_en"])
	prepared_df["title"] = pd.Series(
		[
			merge_non_empty_text(title_value, title_en_value)
			for title_value, title_en_value in zip(
				bilby_title_series,
				bilby_title_en_series,
				strict=False,
			)
		],
		index=prepared_df.index,
		dtype="object",
	)
	prepared_df["source_url"] = coalesce_columns(prepared_df, ["article_url"])
	prepared_df["site_root_url"] = coalesce_columns(prepared_df, ["bilby_site_root_url"])
	prepared_df["published_date"] = coalesce_columns(prepared_df, ["bilby_published_date"])
	prepared_df["summary"] = coalesce_columns(prepared_df, ["bilby_summary"])
	prepared_df["type"] = "bilby"
	return prepared_df


def prepare_gta_unmatched_output_columns(gta_unmatched_df: pd.DataFrame) -> pd.DataFrame:
	prepared_df = gta_unmatched_df.copy()
	main_title_series = coalesce_columns(prepared_df, ["main_title"])
	source_title_series = coalesce_columns(prepared_df, ["source_title"])
	prepared_df["title"] = pd.Series(
		[
			merge_non_empty_text(main_title, source_title)
			for main_title, source_title in zip(main_title_series, source_title_series, strict=False)
		],
		index=prepared_df.index,
		dtype="object",
	)
	prepared_df["source_url"] = coalesce_columns(prepared_df, ["source_url"])
	prepared_df["site_root_url"] = coalesce_columns(prepared_df, ["site_root_url"])
	prepared_df["summary"] = coalesce_columns(prepared_df, ["summary"])
	prepared_df["published_date"] = coalesce_columns(prepared_df, ["published_date"])
	prepared_df["type"] = "gta"

	return prepared_df


def prepare_bilby_matched_output_columns(matched_df: pd.DataFrame) -> pd.DataFrame:
	matched_unique = matched_df.drop_duplicates(subset=["article_url"]).copy()
	prepared_df = prepare_bilby_output_columns(matched_unique)
	prepared_df = prepared_df[BASE_OUTPUT_COLUMNS].copy()
	prepared_df["label"] = 1
	return prepared_df


def prepare_bilby_unmatched_output_columns(bilby_unmatched_df: pd.DataFrame) -> pd.DataFrame:
	prepared_df = prepare_bilby_output_columns(bilby_unmatched_df)
	prepared_df = prepared_df[BASE_OUTPUT_COLUMNS].copy()
	prepared_df["label"] = 0
	return prepared_df


def prepare_gta_label1_output_columns(gta_df: pd.DataFrame) -> pd.DataFrame:
	prepared_df = prepare_gta_unmatched_output_columns(gta_df)
	prepared_df = prepared_df[BASE_OUTPUT_COLUMNS].copy()
	prepared_df["label"] = 1
	return prepared_df


def merge_labeled_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
	if not frames:
		return pd.DataFrame(columns=STANDARD_OUTPUT_COLUMNS)
	combined_df = pd.concat(frames, ignore_index=True, sort=False)
	if "label" not in combined_df.columns:
		combined_df["label"] = 0
	combined_df["label"] = combined_df["label"].fillna(0).astype(int)
	combined_df = combined_df.sort_values("label", ascending=False)
	combined_df = combined_df.drop_duplicates(subset=["source_url"], keep="first")
	return combined_df[STANDARD_OUTPUT_COLUMNS].copy()


def build_base_labeled_dataset(
	matched_df: pd.DataFrame,
	bilby_unmatched_df: pd.DataFrame,
) -> pd.DataFrame:
	label1_df = prepare_bilby_matched_output_columns(matched_df)
	label0_df = prepare_bilby_unmatched_output_columns(bilby_unmatched_df)
	return merge_labeled_frames([label1_df, label0_df])


def build_dataset_from_sources(
	combine_matched_df: pd.DataFrame,
	combine_bilby_unmatched_df: pd.DataFrame,
	use_embedding_source: bool,
	embedding_matched_df: pd.DataFrame | None = None,
	embedding_bilby_unmatched_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if use_embedding_source:
		if embedding_matched_df is None or embedding_bilby_unmatched_df is None:
			raise ValueError("Embedding source frames are required when use_embedding_source=True.")
		selected_matched_df = embedding_matched_df.copy()
		selected_bilby_unmatched_df = embedding_bilby_unmatched_df.copy()
	else:
		selected_matched_df = combine_matched_df.copy()
		selected_bilby_unmatched_df = combine_bilby_unmatched_df.copy()
	dataset_df = build_base_labeled_dataset(selected_matched_df, selected_bilby_unmatched_df)
	return dataset_df, selected_matched_df, selected_bilby_unmatched_df


def clean_root_series(df: pd.DataFrame, column: str) -> pd.Series:
	if column not in df.columns:
		return pd.Series([""] * len(df), index=df.index, dtype="object")
	return df[column].map(clean_root_value)


def build_distribution_table(
	matched_df: pd.DataFrame,
	bilby_unmatched_df: pd.DataFrame,
	gta_frames: list[pd.DataFrame],
) -> pd.DataFrame:
	# Distribution counts deduplicate using article_url (mapped to output source_url)
	# for matched records, and source_url for GTA-only additions.
	gta_source_frames: list[pd.DataFrame] = []
	gta_matched = matched_df.copy()
	gta_matched["site_root_url"] = clean_root_series(gta_matched, "site_root_url")
	gta_matched["source_url"] = clean_root_series(gta_matched, "article_url")
	gta_source_frames.append(gta_matched[["site_root_url", "source_url"]])
	for gta_df in gta_frames:
		extra_df = gta_df.copy()
		extra_df["site_root_url"] = clean_root_series(extra_df, "site_root_url")
		extra_df["source_url"] = clean_root_series(extra_df, "source_url")
		gta_source_frames.append(extra_df[["site_root_url", "source_url"]])
	gta_all = pd.concat(gta_source_frames, ignore_index=True, sort=False)
	gta_all = gta_all.loc[gta_all["site_root_url"].ne("") & gta_all["source_url"].ne("")].copy()
	gta_all = gta_all.drop_duplicates(subset=["source_url"]).copy()
	gta_counts = gta_all.groupby("site_root_url").size().rename("gta_count").reset_index()

	# Bilby distribution: combine matched + bilby unmatched, dedup by article_url.
	bilby_matched = matched_df.copy()
	bilby_matched["site_root_url"] = clean_root_series(bilby_matched, "bilby_site_root_url")
	bilby_matched["article_url"] = clean_root_series(bilby_matched, "article_url")
	bilby_unmatched = bilby_unmatched_df.copy()
	bilby_unmatched["site_root_url"] = clean_root_series(bilby_unmatched, "bilby_site_root_url")
	bilby_unmatched["article_url"] = clean_root_series(bilby_unmatched, "article_url")
	bilby_all = pd.concat(
		[
			bilby_matched[["site_root_url", "article_url"]],
			bilby_unmatched[["site_root_url", "article_url"]],
		],
		ignore_index=True,
		sort=False,
	)
	bilby_all = bilby_all.loc[bilby_all["site_root_url"].ne("") & bilby_all["article_url"].ne("")].copy()
	bilby_all = bilby_all.drop_duplicates(subset=["article_url"]).copy()
	bilby_counts = bilby_all.groupby("site_root_url").size().rename("bilby_count").reset_index()

	# Matched amount: combine matched dedup by article_url.
	matched_amount_df = bilby_matched.loc[
		bilby_matched["site_root_url"].ne("") & bilby_matched["article_url"].ne("")
	].copy()
	matched_amount_df = matched_amount_df.drop_duplicates(subset=["article_url"]).copy()
	matched_counts = (
		matched_amount_df.groupby("site_root_url").size().rename("matched_amount").reset_index()
	)

	table_df = gta_counts.merge(bilby_counts, on="site_root_url", how="inner")
	table_df = table_df.merge(matched_counts, on="site_root_url", how="left")
	table_df["matched_amount"] = table_df["matched_amount"].fillna(0).astype(int)
	table_df = table_df.loc[(table_df["gta_count"] > 0) & (table_df["bilby_count"] > 0)].copy()
	table_df["matched_rate"] = table_df["matched_amount"] / table_df["bilby_count"]
	table_df = table_df.sort_values(["matched_rate", "matched_amount"], ascending=[False, False]).reset_index(drop=True)
	return table_df


def print_distribution_table(table_df: pd.DataFrame, title: str) -> None:
	print(f"\n=== {title} ===")
	if table_df.empty:
		print("No overlapping site_root_url with non-zero GTA and Bilby counts.")
		return
	display_df = table_df.copy()
	display_df["matched_rate"] = display_df["matched_rate"].map(lambda value: f"{value:.2%}")
	print(display_df.to_string(index=False))


def ask_choice(question_text: str, choices: dict[str, str]) -> str:
	while True:
		print(f"\n{question_text}")
		for option, description in choices.items():
			print(f"  {option}: {description}")
		answer = input("Enter choice: ").strip()
		if answer in choices:
			return answer
		valid_choices = ", ".join(choices.keys())
		print(f"Invalid input. Please choose one of: {valid_choices}.")


def append_label1_gta_dataset(base_df: pd.DataFrame, gta_df: pd.DataFrame) -> pd.DataFrame:
	label1_gta_df = prepare_gta_label1_output_columns(gta_df)
	return merge_labeled_frames([base_df, label1_gta_df])


def export_dataset(dataset_df: pd.DataFrame, output_path: Path) -> Path:
	resolved_output = output_path.resolve()
	resolved_output.parent.mkdir(parents=True, exist_ok=True)
	dataset_df.to_parquet(resolved_output, index=False)
	excel_output = resolved_output.with_suffix(".xlsx")
	sanitize_excel_dataframe(dataset_df).to_excel(excel_output, index=False)
	print(f"Parquet exported to: {resolved_output}")
	print(f"Excel exported to: {excel_output}")
	print(f"Rows written: {len(dataset_df)}")
	return resolved_output


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Build a labeled dataset from combine/embedding match outputs with interactive source and GTA expansion choices."
	)
	parser.add_argument(
		"--match-dir",
		type=Path,
		default=DEFAULT_MATCH_OUTPUT_DIR,
		help="Directory containing combine, embedding, and GTA out-of-range parquet files.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional final output parquet path.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory used when --output is not provided.",
	)
	parser.add_argument(
		"--rate-threshold",
		type=float,
		default=0.02,
		help="Keep site_root_url rows with matched_rate >= this threshold.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not 0 <= args.rate_threshold <= 1:
		raise ValueError("--rate-threshold must be between 0 and 1")
	print(
		"\nNote: All distribution tables are sorted by matched_rate in descending order. "
		f"Only source entries with matched_rate >= {args.rate_threshold:.2f} will be included in the final dataset. "
		"Dedup uses article_url (or output source_url) rather than the raw source_url in matched parquet files."
	)

	# Step 1: Load combine and embedding sources, then print both baseline distribution tables.
	combine_matched_df, combine_bilby_unmatched_df = load_combine_parquets(args.match_dir)
	embedding_matched_df, embedding_bilby_unmatched_df = load_embedding_parquets(args.match_dir)

	combine_distribution_df = build_distribution_table(
		matched_df=combine_matched_df,
		bilby_unmatched_df=combine_bilby_unmatched_df,
		gta_frames=[],
	)
	print_distribution_table(combine_distribution_df, "Combine Baseline Distribution")

	embedding_distribution_df = build_distribution_table(
		matched_df=embedding_matched_df,
		bilby_unmatched_df=embedding_bilby_unmatched_df,
		gta_frames=[],
	)
	print_distribution_table(
		embedding_distribution_df,
		"current embedding output + combine result distribution",
	)

	# Step 2: Choose base data source (combine-only vs embedding-only).
	base_choice = ask_choice(
		"combine outputs (url + title) only",
		{
			"0": "Use combine outputs only (exact URL/title branch).",
			"1": "Use embedding + combine match (please make sure that the current embedding output is the file for your required time range).",
		},
	)
	use_embedding_source = base_choice == "1"
	dataset_df, selected_matched_df, selected_bilby_unmatched_df = build_dataset_from_sources(
		combine_matched_df=combine_matched_df,
		combine_bilby_unmatched_df=combine_bilby_unmatched_df,
		use_embedding_source=use_embedding_source,
		embedding_matched_df=embedding_matched_df,
		embedding_bilby_unmatched_df=embedding_bilby_unmatched_df,
	)

	# Step 3: Choose GTA additions to label=1.
	gta_choice = ask_choice(
		"Choose GTA dataset additions for label=1:",
		{
			"0": "Add current GTA unmatched dataset.",
			"1": "Add out-of-period GTA dataset.",
			"2": "Add both GTA unmatched and out-of-period datasets.",
			"3": "Add none; keep current output only.",
		},
	)

	gta_frames_for_distribution: list[pd.DataFrame] = []
	if gta_choice in {"0", "2"}:
		gta_unmatched_df = load_selected_gta_unmatched_parquet(
			args.match_dir,
			use_embedding_source=use_embedding_source,
		)
		dataset_df = append_label1_gta_dataset(dataset_df, gta_unmatched_df)
		gta_frames_for_distribution.append(gta_unmatched_df)

	if gta_choice in {"1", "2"}:
		out_of_period_path = require_parquet_file(
			args.match_dir.resolve() / GTA_NOT_IN_TIME_RANGE_FILENAME,
			"Out-of-period GTA",
		)
		gta_out_of_period_df = pd.read_parquet(out_of_period_path)
		dataset_df = append_label1_gta_dataset(dataset_df, gta_out_of_period_df)
		gta_frames_for_distribution.append(gta_out_of_period_df)

	# Step 4: Recompute and print updated distribution after user selections.
	distribution_df = build_distribution_table(
		matched_df=selected_matched_df,
		bilby_unmatched_df=selected_bilby_unmatched_df,
		gta_frames=gta_frames_for_distribution,
	)
	print_distribution_table(distribution_df, "Final Distribution After Selection")

	# Step 5: Keep only site roots whose matched_rate meets threshold.
	qualified_roots = set(
		distribution_df.loc[
			distribution_df["matched_rate"].ge(args.rate_threshold), "site_root_url"
		].tolist()
	)
	final_dataset_df = dataset_df.loc[
		dataset_df["site_root_url"].map(clean_root_value).isin(qualified_roots)
	].copy()
	final_dataset_df = final_dataset_df.drop_duplicates(subset=["source_url"], keep="first").copy()

	output_path = (
		args.output.resolve()
		if args.output is not None
		else args.output_dir.resolve() / DEFAULT_OUTPUT_FILENAME
	)
	print(f"\nQualified site_root_url count (matched_rate >= {args.rate_threshold:.2f}): {len(qualified_roots)}")
	final_label1_rows = int(final_dataset_df["label"].eq(1).sum()) if not final_dataset_df.empty else 0
	final_label0_rows = int(final_dataset_df["label"].eq(0).sum()) if not final_dataset_df.empty else 0
	total_rows = len(final_dataset_df)
	label1_rate = (final_label1_rows / total_rows) if total_rows else 0.0
	print(f"Final label=1 rows: {final_label1_rows}")
	print(f"Final label=0 rows: {final_label0_rows}")
	print(f"Final total rows: {total_rows}")
	print(f"Final label=1 rate: {label1_rate:.2%}")
	export_dataset(final_dataset_df, output_path)


if __name__ == "__main__":
	main()
