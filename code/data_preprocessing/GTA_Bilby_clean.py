# GTA：
#  - 读取准备好的 GTA 文件
#  - 清洗 site_root_url（english. 前缀归一化）
#  - 保留并打印 published_date 范围与文档总数
#  - 输出清洗后的 GTA 文件
# Bilby：
#  - 读取 Bilby Parquet 文件
#  - 清洗 URL、代理标记与 site_root_url
#  - 保留并打印 published_date 范围与文档总数
#  - 输出清洗后的 Bilby 文件

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import pandas as pd
import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GTA_INPUT = ROOT_DIR / "Data" / "China_GTA_Source" / "interventions_sources.parquet"
DEFAULT_GTA_OUTPUT = ROOT_DIR / "Data" / "China_GTA_Source" / "interventions_sources_cleaned.parquet"
DEFAULT_BILBY_INPUT = ROOT_DIR / "Data" / "Bilby_data_fixed" / "Bilby_2025_2026_combined.parquet"
DEFAULT_BILBY_OUTPUT = ROOT_DIR / "Data" / "Bilby_data_fixed" / "Bilby_2025_2026_combined_cleaned.parquet"
DEFAULT_START_DATE = pd.Timestamp("2025-06-01")
DEFAULT_END_DATE = pd.Timestamp("2026-03-01")
GTA_DATE_COLUMN = "published_date"
GTA_MAIN_TITLE_COLUMN = "main_title"
GTA_ORIGINAL_SITE_ROOT_COLUMN = "original_site_root_url"
GTA_SITE_ROOT_NORMALIZED_COLUMN = "site_root_url_normalized"
BILBY_ORIGINAL_SITE_ROOT_COLUMN = "original_site_root_url"
BILBY_DATE_COLUMN = "published_date_cn"
EXCEL_ILLEGAL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
EXCEL_MAX_SHEET_ROWS = 1_048_576
EXCEL_MAX_DATA_ROWS_PER_SHEET = EXCEL_MAX_SHEET_ROWS - 1
BILBY_EXPORT_DROP_COLUMNS = [
	"uuid",
	"branch_id",
	"translated_summary",
	"published_at",
	"published_at_cn",
	"subhead",
	"subhead_en",
	"author",
	"news_line",
	"newspaper"
]
GTA_TITLE_CLEAN_SPLIT_RE = re.compile(
	r"\b(?:retrieved(?:\s+on)?|avaliable\s+at|available\s+at|new\s+release|press\s+release|published\s+on)\b",
	flags=re.IGNORECASE,
)


def normalize_url(url: str | None) -> str | None:
	if url is None:
		return None

	clean_url = url.strip()
	if not clean_url:
		return None
	if clean_url.startswith("//"):
		clean_url = f"https:{clean_url}"

	return clean_url


def is_scrapeops_proxy_url(url: str | None) -> bool:
	clean_url = normalize_url(url)
	if clean_url is None:
		return False

	parsed_url = urlparse(clean_url)
	return parsed_url.netloc.lower() == "proxy.scrapeops.io"


def resolve_article_url(url: str | None) -> str | None:
	clean_url = normalize_url(url)
	if clean_url is None:
		return None

	parsed_url = urlparse(clean_url)
	if parsed_url.netloc.lower() != "proxy.scrapeops.io":
		return clean_url

	target_url = parse_qs(parsed_url.query).get("url", [None])[0]
	resolved_url = normalize_url(unquote(target_url) if target_url else None)
	return resolved_url or clean_url


def resolve_site_root_url(url: str | None) -> str | None:
	clean_url = normalize_url(url)
	if not clean_url:
		return None

	parsed_url = urlparse(clean_url)
	if not parsed_url.scheme or not parsed_url.netloc:
		return None

	return urlunparse((parsed_url.scheme, parsed_url.netloc, "/", "", "", ""))


def extract_url_host(url: str | None) -> str | None:
	clean_url = normalize_url(url)
	if not clean_url:
		return None

	parsed_url = urlparse(clean_url)
	if not parsed_url.netloc:
		return None

	return parsed_url.netloc


def normalize_english_site_root_url(url: str | None) -> str | None:
	clean_url = normalize_url(url)
	if not clean_url:
		return clean_url

	parsed_url = urlparse(clean_url)
	host = parsed_url.netloc
	if not parsed_url.scheme or not host:
		return clean_url
	if not host.lower().startswith("english."):
		return clean_url

	stripped_host = host[8:]
	normalized_host = stripped_host if stripped_host.lower().startswith("www.") else f"www.{stripped_host}"
	return urlunparse(
		(
			parsed_url.scheme,
			normalized_host,
			parsed_url.path,
			parsed_url.params,
			parsed_url.query,
			parsed_url.fragment,
		)
	)


def resolve_input_path(primary_path: Path, fallback_paths: list[Path]) -> Path:
	for candidate in [primary_path, *fallback_paths]:
		if candidate.exists():
			return candidate.resolve()
	raise FileNotFoundError(f"Input file not found. Tried: {[str(path) for path in [primary_path, *fallback_paths]]}")


def load_gta_data(input_path: Path) -> pd.DataFrame:
	if input_path.suffix.lower() == ".parquet":
		df = pl.read_parquet(input_path).to_pandas()
		if GTA_DATE_COLUMN not in df.columns:
			raise KeyError(f"Required column not found: {GTA_DATE_COLUMN}")
	else:
		excel_file = pd.ExcelFile(input_path)
		df: pd.DataFrame | None = None
		for sheet_name in excel_file.sheet_names:
			candidate_df = pd.read_excel(excel_file, sheet_name=sheet_name, dtype={GTA_DATE_COLUMN: "string"})
			if GTA_DATE_COLUMN in candidate_df.columns:
				df = candidate_df
				break

		if df is None:
			raise KeyError(
				f"Required column not found: {GTA_DATE_COLUMN}. Checked sheets: {excel_file.sheet_names}"
			)

	if "site_root_url" in df.columns:
		original_site_root_url = df["site_root_url"].copy()
		df[GTA_ORIGINAL_SITE_ROOT_COLUMN] = original_site_root_url
		normalized_site_root_url = df["site_root_url"].map(normalize_english_site_root_url)
		df[GTA_SITE_ROOT_NORMALIZED_COLUMN] = normalized_site_root_url.ne(original_site_root_url)
		df["site_root_url"] = normalized_site_root_url.map(extract_url_host)
	else:
		df[GTA_ORIGINAL_SITE_ROOT_COLUMN] = None
		df[GTA_SITE_ROOT_NORMALIZED_COLUMN] = False

	df[GTA_DATE_COLUMN] = pd.to_datetime(df[GTA_DATE_COLUMN], errors="coerce")
	return df


def clean_gta_title_text(value: object) -> object:
	if pd.isna(value):
		return value
	text = str(value)
	prefix = GTA_TITLE_CLEAN_SPLIT_RE.split(text, maxsplit=1)[0]
	prefix = re.sub(r"\s+", " ", prefix).strip(" \t\r\n,;:()[]{}-")
	return prefix


def clean_gta_data(gta_df: pd.DataFrame) -> pd.DataFrame:
	# Keep GTA cleaning only; no published-date window filtering.
	cleaned_df = gta_df.copy()
	for title_column in ["source_title", "gta_title", "title", GTA_MAIN_TITLE_COLUMN]:
		if title_column in cleaned_df.columns:
			cleaned_df[title_column] = cleaned_df[title_column].map(clean_gta_title_text)
	return cleaned_df


def enrich_gta_main_title(gta_df: pd.DataFrame, lookup_input_path: Path) -> pd.DataFrame:
	if GTA_MAIN_TITLE_COLUMN in gta_df.columns:
		return gta_df

	if not lookup_input_path.exists():
		print(f"Warning: main_title lookup file not found: {lookup_input_path}")
		gta_df[GTA_MAIN_TITLE_COLUMN] = ""
		return gta_df

	lookup_df = pl.read_parquet(lookup_input_path).to_pandas()
	if GTA_MAIN_TITLE_COLUMN not in lookup_df.columns:
		print(f"Warning: '{GTA_MAIN_TITLE_COLUMN}' not found in lookup file: {lookup_input_path}")
		gta_df[GTA_MAIN_TITLE_COLUMN] = ""
		return gta_df

	# Prefer source_url for one-to-one title mapping; fallback to state_act_url when needed.
	for key_column in ["source_url", "state_act_url"]:
		if key_column in gta_df.columns and key_column in lookup_df.columns:
			lookup = (
				lookup_df[[key_column, GTA_MAIN_TITLE_COLUMN]]
				.dropna(subset=[key_column])
				.drop_duplicates(subset=[key_column], keep="first")
			)
			merged = gta_df.merge(lookup, on=key_column, how="left")
			if GTA_MAIN_TITLE_COLUMN in merged.columns:
				merged[GTA_MAIN_TITLE_COLUMN] = merged[GTA_MAIN_TITLE_COLUMN].fillna("")
				return merged

	gta_df[GTA_MAIN_TITLE_COLUMN] = ""
	print("Warning: unable to map main_title (no shared key column found).")
	return gta_df


def load_bilby_data(input_path: Path) -> pl.DataFrame:
	bilby_df = pl.read_parquet(input_path)
	if "article_url" not in bilby_df.columns:
		raise KeyError("Required column not found: article_url")
	return bilby_df.unique(subset=["article_url"], keep="first", maintain_order=True)


def clean_bilby_data(bilby_df: pl.DataFrame) -> pl.DataFrame:
	original_url_expr = (
		pl.col("original_article_url")
		if "original_article_url" in bilby_df.columns
		else pl.col("article_url")
	)
	original_site_root_expr = (
		pl.col(BILBY_ORIGINAL_SITE_ROOT_COLUMN)
		if BILBY_ORIGINAL_SITE_ROOT_COLUMN in bilby_df.columns
		else (
			pl.col("site_root_url")
			if "site_root_url" in bilby_df.columns
			else pl.col("article_url").map_elements(resolve_site_root_url, return_dtype=pl.String)
		)
	)
	article_url_expr = pl.col("article_url")
	article_url_is_proxy_expr = (
		pl.col("article_url_is_proxy")
		if "article_url_is_proxy" in bilby_df.columns
		else pl.col("article_url")
		.map_elements(is_scrapeops_proxy_url, return_dtype=pl.Boolean)
	)
	site_root_url_expr = (
		pl.col("site_root_url")
		if "site_root_url" in bilby_df.columns
		else pl.col("article_url")
		.map_elements(resolve_site_root_url, return_dtype=pl.String)
	)

	return bilby_df.with_columns(
		pl.col("published_at").dt.convert_time_zone("Asia/Shanghai").alias("published_at_cn")
	).with_columns(
		original_url_expr.alias("original_article_url"),
		original_site_root_expr.alias(BILBY_ORIGINAL_SITE_ROOT_COLUMN),
		article_url_is_proxy_expr.alias("article_url_is_proxy"),
		article_url_expr.alias("article_url"),
	).with_columns(
		pl.col("published_at_cn").dt.date().alias(BILBY_DATE_COLUMN),
		site_root_url_expr.map_elements(extract_url_host, return_dtype=pl.String).alias("site_root_url"),
	)


def derive_parquet_output_path(output_path: Path) -> Path:
	return output_path.with_suffix(".parquet")


def derive_excel_output_path(output_path: Path) -> Path:
	return output_path.with_suffix(".xlsx")


def write_excel_with_row_split(df: pd.DataFrame, output_path: Path) -> None:
	if len(df) <= EXCEL_MAX_DATA_ROWS_PER_SHEET:
		df.to_excel(output_path, index=False)
		return

	print(
		f"Warning: row count {len(df):,} exceeds Excel single-sheet limit "
		f"({EXCEL_MAX_DATA_ROWS_PER_SHEET:,}). Exporting across multiple sheets."
	)
	with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
		for sheet_index, start in enumerate(range(0, len(df), EXCEL_MAX_DATA_ROWS_PER_SHEET), start=1):
			chunk = df.iloc[start : start + EXCEL_MAX_DATA_ROWS_PER_SHEET]
			chunk.to_excel(writer, sheet_name=f"data_{sheet_index}", index=False)


def export_bilby_data(filtered_df: pl.DataFrame, output_path: Path) -> Path:
	resolved_output = output_path.resolve()
	resolved_parquet_output = derive_parquet_output_path(resolved_output)
	resolved_excel_output = derive_excel_output_path(resolved_output)
	resolved_output.parent.mkdir(parents=True, exist_ok=True)
	export_df = prepare_bilby_export_dataframe(filtered_df)
	pl.from_pandas(export_df).write_parquet(resolved_parquet_output, compression="zstd")
	write_excel_with_row_split(export_df, resolved_excel_output)
	print(f"Filtered Bilby parquet exported to: {resolved_parquet_output}")
	print(f"Filtered Bilby Excel exported to: {resolved_excel_output}")
	return resolved_output


def export_gta_data(filtered_df: pd.DataFrame, output_path: Path) -> Path:
	resolved_output = output_path.resolve()
	resolved_parquet_output = derive_parquet_output_path(resolved_output)
	resolved_excel_output = derive_excel_output_path(resolved_output)
	resolved_output.parent.mkdir(parents=True, exist_ok=True)
	export_df = prepare_excel_dataframe(filtered_df)
	pl.from_pandas(export_df).write_parquet(resolved_parquet_output, compression="zstd")
	write_excel_with_row_split(export_df, resolved_excel_output)
	print(f"Filtered GTA parquet exported to: {resolved_parquet_output}")
	print(f"Filtered GTA Excel exported to: {resolved_excel_output}")
	return resolved_output


def prepare_excel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	excel_df = df.copy()
	for column in excel_df.columns:
		column_data = excel_df[column]
		if isinstance(column_data.dtype, pd.DatetimeTZDtype):
			excel_df[column] = column_data.dt.tz_localize(None)
		elif pd.api.types.is_string_dtype(column_data.dtype) or column_data.dtype == object:
			excel_df[column] = column_data.map(
				lambda value: EXCEL_ILLEGAL_CHARACTERS_RE.sub("", value)
				if isinstance(value, str)
				else value
			)
	return excel_df


def prepare_bilby_export_dataframe(filtered_df: pl.DataFrame) -> pd.DataFrame:
	export_df = filtered_df.drop(
		[column for column in BILBY_EXPORT_DROP_COLUMNS if column in filtered_df.columns]
	)
	return prepare_excel_dataframe(export_df.to_pandas())


def print_gta_summary(filtered_df: pd.DataFrame) -> None:
	print("=== GTA Data Summary ===")
	if filtered_df.empty:
		print("Published date range: no rows")
	else:
		valid_dates = filtered_df[GTA_DATE_COLUMN].dropna()
		if valid_dates.empty:
			print("Published date range: all values are null")
		else:
			print(f"Published date range: {valid_dates.min().date()} to {valid_dates.max().date()}")

	print(f"Total documents: {len(filtered_df):,}")
	print(f"Column count: {len(filtered_df.columns)}")
	print("Columns:")
	for column in filtered_df.columns:
		print(f"- {column}")


def print_bilby_summary(filtered_df: pl.DataFrame) -> None:
	print("\n=== Bilby Data Summary ===")
	if filtered_df.is_empty():
		print("Published date range: no rows")
	else:
		date_range = filtered_df.select(
			pl.col(BILBY_DATE_COLUMN).min().alias("min_date"),
			pl.col(BILBY_DATE_COLUMN).max().alias("max_date"),
		).row(0, named=True)
		print(f"Published date range: {date_range['min_date']} to {date_range['max_date']}")

	print(f"Total documents: {filtered_df.height:,}")
	print(f"Column count: {filtered_df.width}")
	print("Columns:")
	for column in filtered_df.columns:
		print(f"- {column}")
	print("First 5 rows:")
	if filtered_df.is_empty():
		print("No data")
	else:
		print(filtered_df.head(5))


def build_monthly_comparison(
	gta_filtered: pd.DataFrame,
	bilby_filtered: pl.DataFrame,
) -> pd.DataFrame:
	if gta_filtered.empty:
		gta_monthly = pd.DataFrame(columns=["month", "gta_count"])
	else:
		gta_monthly = (
			gta_filtered.assign(
				month=pd.to_datetime(gta_filtered[GTA_DATE_COLUMN]).dt.to_period("M").astype(str)
			)
			.groupby("month", as_index=False)
			.size()
			.rename(columns={"size": "gta_count"})
		)

	if bilby_filtered.is_empty():
		bilby_monthly = pd.DataFrame(columns=["month", "bilby_count"])
	else:
		bilby_monthly = (
			bilby_filtered.with_columns(
				pl.col(BILBY_DATE_COLUMN).dt.strftime("%Y-%m").alias("month")
			)
			.group_by("month")
			.len()
			.rename({"len": "bilby_count"})
			.sort("month")
			.to_pandas()
		)

	comparison = (
		gta_monthly.merge(bilby_monthly, on="month", how="outer")
		.fillna(0)
		.sort_values("month")
	)

	if comparison.empty:
		return pd.DataFrame(columns=["month", "gta_count", "bilby_count", "difference"])

	comparison[["gta_count", "bilby_count"]] = comparison[["gta_count", "bilby_count"]].astype(int)
	comparison["difference"] = comparison["bilby_count"] - comparison["gta_count"]
	return comparison


def print_monthly_comparison(comparison_df: pd.DataFrame) -> None:
	print("\n=== Monthly GTA vs Bilby Counts ===")
	if comparison_df.empty:
		print("No data")
		return
	print(comparison_df.to_string(index=False))


def run_gta_bilby_read(
	gta_input_path: Path,
	bilby_input_path: Path,
	start_date: pd.Timestamp = DEFAULT_START_DATE,
	end_date: pd.Timestamp = DEFAULT_END_DATE,
	gta_output_path: Path = DEFAULT_GTA_OUTPUT,
	bilby_output_path: Path = DEFAULT_BILBY_OUTPUT,
) -> tuple[pd.DataFrame, pl.DataFrame, pd.DataFrame]:
	resolved_gta_input = resolve_input_path(gta_input_path, [])
	resolved_bilby_input = resolve_input_path(
		bilby_input_path,
		[
			ROOT_DIR / "Data" / "Bilby_data_fixed" / "Bilby_2025_2026_combined.parquet",
			ROOT_DIR / "Bilby_data" / "Bilby_2025_2026_combined.parquet",
			ROOT_DIR / "Bilby" / "Bilby_2025_2026_combined.parquet",
			ROOT_DIR / "Bilby_2025_2026_combined.parquet",
		],
	)

	gta_df = load_gta_data(resolved_gta_input)
	bilby_df = load_bilby_data(resolved_bilby_input)

	gta_filtered = clean_gta_data(gta_df)
	gta_filtered = enrich_gta_main_title(gta_filtered, DEFAULT_GTA_INPUT)
	bilby_filtered = clean_bilby_data(bilby_df)
	comparison_df = pd.DataFrame(columns=["month", "gta_count", "bilby_count", "difference"])

	print_gta_summary(gta_filtered)
	print_bilby_summary(bilby_filtered)
	export_gta_data(gta_filtered, gta_output_path)
	export_bilby_data(bilby_filtered, bilby_output_path)

	return gta_filtered, bilby_filtered, comparison_df


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Clean GTA and Bilby data without published-date filtering, then print date range and doc counts."
	)
	parser.add_argument(
		"--gta-input",
		type=Path,
		default=DEFAULT_GTA_INPUT,
		help="Prepared GTA input file (parquet or Excel).",
	)
	parser.add_argument(
		"--gta-output",
		type=Path,
		default=DEFAULT_GTA_OUTPUT,
		help="Output path for the filtered GTA data (.parquet or .xlsx).",
	)
	parser.add_argument(
		"--bilby-input",
		type=Path,
		default=DEFAULT_BILBY_INPUT,
		help="Combined Bilby parquet file.",
	)
	parser.add_argument(
		"--start-date",
		type=str,
		default=str(DEFAULT_START_DATE.date()),
		help="Unused in current cleaning-only mode (kept for backward compatibility).",
	)
	parser.add_argument(
		"--end-date",
		type=str,
		default=str(DEFAULT_END_DATE.date()),
		help="Unused in current cleaning-only mode (kept for backward compatibility).",
	)
	parser.add_argument(
		"--bilby-output",
		type=Path,
		default=DEFAULT_BILBY_OUTPUT,
		help="Output path for the filtered Bilby data (.parquet or .xlsx).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_gta_bilby_read(
		gta_input_path=args.gta_input,
		bilby_input_path=args.bilby_input,
		start_date=pd.Timestamp(args.start_date),
		end_date=pd.Timestamp(args.end_date),
		gta_output_path=args.gta_output,
		bilby_output_path=args.bilby_output,
	)


if __name__ == "__main__":
	main()
