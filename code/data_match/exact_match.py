from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pandas as pd
import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GTA_INPUT = ROOT_DIR / "Data" / "China_GTA_Source" / "interventions_sources_cleaned.parquet"
DEFAULT_BILBY_INPUT = ROOT_DIR / "Data" / "Bilby_data_fixed" / "Bilby_2025_2026_combined_cleaned.parquet"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Data" / "match_outputs"
DEFAULT_START_DATE = pd.Timestamp("2025-06-01")
DEFAULT_END_DATE = pd.Timestamp("2026-03-01")
ILLEGAL_EXCEL_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
ENGLISH_ONLY_PATTERN = re.compile(r"^[A-Za-z0-9\s\.,;:'\"()\-—/&\[\]!\?\+%]+$")


def clean_text(value: object) -> str:
	if pd.isna(value):
		return ""
	return str(value).strip()


def normalize_title_for_exact_match(value: object) -> str:
	text = clean_text(value)
	if not text:
		return ""

	# Keep only letters and digits so spaces and punctuation do not affect exact title matching.
	return "".join(character.casefold() for character in text if character.isalnum())


def normalize_english_title_for_exact_match(value: object) -> str:
	text = clean_text(value)
	if not text:
		return ""
	# English title match: lowercase and remove spaces/punctuation by keeping only letters/digits.
	return "".join(character.casefold() for character in text if character.isalnum())


def normalize_non_english_title_for_exact_match(value: object) -> str:
	text = clean_text(value)
	if not text:
		return ""
	# Chinese/mixed title match: lowercase and remove spaces/punctuation by keeping only alnum chars.
	return "".join(character.casefold() for character in text if character.isalnum())


def normalize_compare_url(value: object) -> str:
	text = clean_text(value)
	if not text:
		return ""
	return re.sub(r"^https?://", "", text, flags=re.IGNORECASE)


def classify_title_lang(value: object) -> str | None:
	text = clean_text(value)
	if not text:
		return None
	return "en" if ENGLISH_ONLY_PATTERN.fullmatch(text) else "zh"


def compute_date_diff_days(gta_published_date: object, bilby_published_date: object) -> float | None:
	if pd.isna(gta_published_date) or pd.isna(bilby_published_date):
		return None
	return float(abs((pd.Timestamp(gta_published_date) - pd.Timestamp(bilby_published_date)).days))


def is_within_one_month(gta_published_date: object, bilby_published_date: object) -> bool:
	if pd.isna(gta_published_date) or pd.isna(bilby_published_date):
		return False
	gta_timestamp = pd.Timestamp(gta_published_date)
	bilby_timestamp = pd.Timestamp(bilby_published_date)
	window_start = gta_timestamp - pd.DateOffset(months=1)
	window_end = gta_timestamp + pd.DateOffset(months=1)
	return window_start <= bilby_timestamp <= window_end


def safe_accuracy(matched_count: int, total_count: int) -> float:
	if total_count == 0:
		return 0.0
	return matched_count / total_count


def ensure_output_dir(output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	return output_dir.resolve()


def resolve_input_path(primary_path: Path, fallback_paths: list[Path]) -> Path:
	for candidate in [primary_path, *fallback_paths]:
		if candidate.exists():
			return candidate.resolve()
	raise FileNotFoundError(f"Input file not found. Tried: {[str(path) for path in [primary_path, *fallback_paths]]}")


def derive_sheet_parquet_path(output_path: Path, sheet_name: str) -> Path:
	return output_path.with_name(f"{output_path.stem}_{sheet_name}.parquet")


def export_match_parquet_files(
	matched_df: pd.DataFrame,
	gta_unmatched_df: pd.DataFrame,
	bilby_unmatched_df: pd.DataFrame,
	output_path: Path,
) -> dict[str, Path]:
	parquet_paths = {
		"matched": derive_sheet_parquet_path(output_path, "matched"),
		"gta_unmatched": derive_sheet_parquet_path(output_path, "gta_unmatched"),
		"bilby_unmatched": derive_sheet_parquet_path(output_path, "bilby_unmatched"),
	}
	matched_df.to_parquet(parquet_paths["matched"], index=False)
	gta_unmatched_df.to_parquet(parquet_paths["gta_unmatched"], index=False)
	bilby_unmatched_df.to_parquet(parquet_paths["bilby_unmatched"], index=False)
	return parquet_paths


def export_match_workbook(
	matched_df: pd.DataFrame,
	gta_unmatched_df: pd.DataFrame,
	bilby_unmatched_df: pd.DataFrame,
	output_path: Path,
) -> dict[str, Path]:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with pd.ExcelWriter(output_path) as writer:
		sanitize_excel_dataframe(matched_df).to_excel(writer, sheet_name="matched", index=False)
		sanitize_excel_dataframe(gta_unmatched_df).to_excel(writer, sheet_name="gta_unmatched", index=False)
		sanitize_excel_dataframe(bilby_unmatched_df).to_excel(writer, sheet_name="bilby_unmatched", index=False)
	return export_match_parquet_files(
		matched_df,
		gta_unmatched_df,
		bilby_unmatched_df,
		output_path,
	)


def sanitize_excel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	sanitized = df.copy()
	for column in sanitized.select_dtypes(include=["object", "string"]).columns:
		sanitized[column] = sanitized[column].map(
			lambda value: ILLEGAL_EXCEL_CHAR_RE.sub("", value)
			if isinstance(value, str)
			else value
		)
	return sanitized


def export_single_sheet_workbook(df: pd.DataFrame, sheet_name: str, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with pd.ExcelWriter(output_path) as writer:
		sanitize_excel_dataframe(df).to_excel(writer, sheet_name=sheet_name, index=False)


def normalize_gta_dates(gta_df: pd.DataFrame) -> pd.DataFrame:
	prepared = gta_df.copy()
	if "published_date" in prepared.columns:
		prepared["__published_date_ts"] = pd.to_datetime(prepared["published_date"], errors="coerce")
	else:
		prepared["__published_date_ts"] = pd.NaT
	return prepared


def normalize_bilby_dates(bilby_df: pl.DataFrame) -> pl.DataFrame:
	if "published_date_cn" not in bilby_df.columns:
		return bilby_df.with_columns(pl.lit(None).cast(pl.Date).alias("__published_date_cn_date"))
	return bilby_df.with_columns(
		pl.col("published_date_cn").cast(pl.Date, strict=False).alias("__published_date_cn_date")
	)


def summarize_dataset_dates(gta_df: pd.DataFrame, bilby_df: pl.DataFrame) -> None:
	gta_total = len(gta_df)
	bilby_total = bilby_df.height

	gta_dates = gta_df["__published_date_ts"].dropna()
	if gta_dates.empty:
		gta_range_text = "No valid published_date values"
	else:
		gta_range_text = f"{gta_dates.min().date()} to {gta_dates.max().date()}"

	bilby_min = bilby_df.select(pl.col("__published_date_cn_date").min()).item()
	bilby_max = bilby_df.select(pl.col("__published_date_cn_date").max()).item()
	if bilby_min is None or bilby_max is None:
		bilby_range_text = "No valid published_date_cn values"
	else:
		bilby_range_text = f"{bilby_min} to {bilby_max}"

	print("\n=== Input Date Coverage ===")
	print(f"GTA total docs: {gta_total}")
	print(f"GTA published_date range: {gta_range_text}")
	print(f"Bilby total docs: {bilby_total}")
	print(f"Bilby published_date_cn range: {bilby_range_text}")


def prompt_date_value(prompt_text: str, default_value: pd.Timestamp) -> pd.Timestamp:
	while True:
		user_input = input(f"{prompt_text} [{default_value.date()}]: ").strip()
		if not user_input:
			return default_value
		try:
			return pd.Timestamp(user_input)
		except (ValueError, TypeError):
			print("Invalid date format. Please use YYYY-MM-DD.")


def resolve_match_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
	default_start = pd.Timestamp(start_date)
	default_end = pd.Timestamp(end_date)

	if not sys.stdin.isatty():
		if default_start >= default_end:
			raise ValueError("start_date must be earlier than end_date.")
		return default_start, default_end

	print("\nEnter desired match date range (start inclusive, end exclusive).")
	while True:
		selected_start = prompt_date_value("Start date (YYYY-MM-DD)", default_start)
		selected_end = prompt_date_value("End date (YYYY-MM-DD)", default_end)
		if selected_start < selected_end:
			return selected_start, selected_end
		print("Invalid range. Start date must be earlier than end date.")


def export_out_of_range_data(
	gta_out_of_range: pd.DataFrame,
	bilby_out_of_range: pl.DataFrame,
	output_dir: Path,
) -> dict[str, Path]:
	resolved_output_dir = ensure_output_dir(output_dir)

	gta_output_path = resolved_output_dir / "gta_not_in_time_range.xlsx"
	gta_sheet = "gta_not_in_time_range"
	gta_export_df = gta_out_of_range.drop(columns=["__published_date_ts"], errors="ignore")
	export_single_sheet_workbook(gta_export_df, gta_sheet, gta_output_path)
	gta_parquet_path = derive_sheet_parquet_path(gta_output_path, gta_sheet)
	gta_export_df.to_parquet(gta_parquet_path, index=False)

	bilby_output_path = resolved_output_dir / "bilby_not_in_time_range.xlsx"
	bilby_sheet = "bilby_not_in_time_range"
	bilby_export_df = bilby_out_of_range.drop("__published_date_cn_date", strict=False).to_pandas()
	export_single_sheet_workbook(bilby_export_df, bilby_sheet, bilby_output_path)
	bilby_parquet_path = derive_sheet_parquet_path(bilby_output_path, bilby_sheet)
	bilby_export_df.to_parquet(bilby_parquet_path, index=False)

	print("\n=== Out-of-Range Exports ===")
	print(f"GTA not in time range rows: {len(gta_export_df)}")
	print(f"Workbook exported to: {gta_output_path}")
	print(f"Parquet exported to: {gta_parquet_path}")
	print(f"Bilby not in time range rows: {len(bilby_export_df)}")
	print(f"Workbook exported to: {bilby_output_path}")
	print(f"Parquet exported to: {bilby_parquet_path}")

	return {
		"gta_not_in_time_range_xlsx": gta_output_path,
		"gta_not_in_time_range_parquet": gta_parquet_path,
		"bilby_not_in_time_range_xlsx": bilby_output_path,
		"bilby_not_in_time_range_parquet": bilby_parquet_path,
	}


def build_combined_match_export(
	url_matched_df: pd.DataFrame,
	title_matched_df: pd.DataFrame,
) -> pd.DataFrame:
	url_export = url_matched_df.copy()
	url_export.insert(0, "match_type", "url_match")
	url_export["matched_title"] = (
		url_export.get("bilby_title", pd.Series(index=url_export.index, dtype="object"))
		.fillna(url_export.get("title", pd.Series(index=url_export.index, dtype="object")))
	)

	title_export = title_matched_df.copy()
	title_export.insert(0, "match_type", "title_match")
	title_export["matched_title"] = (
		title_export.get("gta_title", pd.Series(index=title_export.index, dtype="object"))
		.fillna(title_export.get("source_title", pd.Series(index=title_export.index, dtype="object")))
	)

	combined_df = pd.concat([url_export, title_export], ignore_index=True, sort=False)
	return combined_df


def run_combined_match(
	gta_filtered: pd.DataFrame,
	bilby_filtered: pl.DataFrame,
	url_matched_df: pd.DataFrame,
	title_matched_df: pd.DataFrame,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
	matched_export_columns = [
		"match_type",
		"matched_title",
		"state_act_url",
		"gta_title",
		"main_title",
		"source_url",
		"site_root_url",
		"published_date",
		"summary",
		"article_url",
		"bilby_site_root_url",
		"bilby_published_date",
		"bilby_title",
		"bilby_title_en",
		"bilby_summary",
	]
	gta_unmatched_columns = [
		"state_act_url",
		"gta_title",
		"main_title",
		"source_url",
		"site_root_url",
		"published_date",
		"summary",
	]
	bilby_unmatched_columns = [
		"article_url",
		"bilby_site_root_url",
		"bilby_published_date",
		"bilby_title",
		"bilby_title_en",
		"bilby_summary",
	]

	def coalesce_columns(formatted: pd.DataFrame, candidates: list[str], fallback: object = pd.NA) -> pd.Series:
		if not candidates:
			return pd.Series([fallback] * len(formatted), index=formatted.index)

		series = pd.Series([pd.NA] * len(formatted), index=formatted.index, dtype="object")
		for candidate in candidates:
			if candidate in formatted.columns:
				series = series.combine_first(formatted[candidate])
		return series.fillna(fallback)

	def with_unified_titles_and_dates(df: pd.DataFrame) -> pd.DataFrame:
		formatted = df.copy()
		formatted["state_act_url"] = coalesce_columns(formatted, ["state_act_url", "state_act_id"])
		formatted["gta_title"] = coalesce_columns(
			formatted,
			["gta_title", "source_title"],
		)
		formatted["main_title"] = coalesce_columns(
			formatted,
			["main_title", "gta_title", "source_title"],
		)
		formatted["gta_title_normalized"] = coalesce_columns(
			formatted,
			["source_title_normalized", "gta_title_normalized"],
		)

		if formatted["gta_title_normalized"].eq("").all() and "source_title" in formatted.columns:
			formatted["gta_title_normalized"] = formatted["source_title"].map(normalize_title_for_exact_match)

		formatted["bilby_published_date"] = coalesce_columns(
			formatted,
			["bilby_published_date", "bilby_published_date_cn"],
		)
		formatted["bilby_title_normalized"] = coalesce_columns(
			formatted,
			["bilby_title_normalized", "bilby_match_title_normalized"],
		)
		formatted["bilby_title"] = coalesce_columns(
			formatted,
			["bilby_title", "title"],
		)
		formatted["bilby_title_normalized"] = formatted["bilby_title_normalized"].replace("", pd.NA)
		formatted["bilby_title_normalized"] = formatted["bilby_title_normalized"].fillna(
			formatted["bilby_title"]
		)
		formatted["bilby_title_en"] = coalesce_columns(
			formatted,
			["bilby_title_en", "title_en"],
		)

		return formatted

	def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
		formatted = with_unified_titles_and_dates(df)
		for column in columns:
			if column not in formatted.columns:
				formatted[column] = pd.NA
		return formatted[columns]

	resolved_output_dir = ensure_output_dir(output_dir)
	combined_matched_export = build_combined_match_export(url_matched_df, title_matched_df)

	gta_base = gta_filtered.copy().reset_index(drop=True)
	gta_base["gta_row_id"] = gta_base.index
	gta_base["gta_title_normalized"] = gta_base["source_title"].map(
		normalize_title_for_exact_match
	)
	bilby_docs = build_bilby_url_lookup(bilby_filtered)
	total_bilby_docs = len(bilby_docs)

	matched_row_ids = set(
		combined_matched_export.get("gta_row_id", pd.Series(dtype="int64")).dropna().astype(int).tolist()
	)
	matched_article_urls = set(
		combined_matched_export.get("article_url", pd.Series(dtype="object")).dropna().astype(str).tolist()
	)

	gta_unmatched_export = gta_base.loc[~gta_base["gta_row_id"].isin(matched_row_ids)].copy()
	bilby_unmatched_export = bilby_docs.loc[
		~bilby_docs["article_url"].astype(str).isin(matched_article_urls)
	].copy()

	combined_matched_export = select_columns(combined_matched_export, matched_export_columns)
	gta_unmatched_export = select_columns(gta_unmatched_export, gta_unmatched_columns)
	bilby_unmatched_export = select_columns(bilby_unmatched_export, bilby_unmatched_columns)

	output_path = resolved_output_dir / "combined_match.xlsx"
	parquet_paths = export_match_workbook(
		combined_matched_export,
		gta_unmatched_export,
		bilby_unmatched_export,
		output_path,
	)

	matched_count = len(matched_row_ids)
	total_count = len(gta_base)
	unique_matched_bilby_count = total_bilby_docs - len(bilby_unmatched_export)
	unmatched_gta_count = total_count - matched_count
	unmatched_bilby_count = len(bilby_unmatched_export)

	print("\n=== Combined Match ===")
	print(f"GTA total rows: {total_count}")
	print(f"Matched by Bilby URL or title/title_en: {matched_count}")
	print(f"Unmatched GTA rows: {unmatched_gta_count}")
	print(f"GTA coverage: {safe_accuracy(matched_count, total_count):.2%}")
	print(f"Unique matched Bilby docs: {unique_matched_bilby_count}")
	print(f"Unmatched Bilby docs: {unmatched_bilby_count}")
	print(f"Bilby coverage: {safe_accuracy(unique_matched_bilby_count, total_bilby_docs):.2%}")
	print(f"Workbook exported to: {output_path}")
	print(f"Parquet exported to: {parquet_paths['matched']}")
	print(f"Parquet exported to: {parquet_paths['gta_unmatched']}")
	print(f"Parquet exported to: {parquet_paths['bilby_unmatched']}")

	return combined_matched_export


def resolve_site_root_url(url: object) -> str | None:
	if pd.isna(url):
		return None

	clean_url = str(url).strip()
	if not clean_url:
		return None
	if clean_url.startswith("//"):
		clean_url = f"https:{clean_url}"

	parsed_url = urlparse(clean_url)
	if not parsed_url.scheme or not parsed_url.netloc:
		return None

	return urlunparse((parsed_url.scheme, parsed_url.netloc, "/", "", "", ""))


def ensure_bilby_site_root_url(bilby_filtered: pl.DataFrame) -> pl.DataFrame:
	if "site_root_url" in bilby_filtered.columns:
		return bilby_filtered

	return bilby_filtered.with_columns(
		pl.col("article_url")
		.map_elements(resolve_site_root_url, return_dtype=pl.String)
		.alias("site_root_url")
	)


def build_bilby_url_lookup(bilby_filtered: pl.DataFrame) -> pd.DataFrame:
	bilby_filtered = ensure_bilby_site_root_url(bilby_filtered)
	lookup = (
		bilby_filtered.select([
			"article_url",
			"original_article_url",
			"site_root_url",
			"original_site_root_url",
			"published_date_cn",
			"title",
			"title_en",
			"summary",
		])
		.to_pandas()
		.drop_duplicates(subset=["article_url"])
	)
	lookup["article_url_compare"] = lookup["article_url"].map(normalize_compare_url)
	lookup["bilby_title_normalized"] = lookup["title"].map(normalize_non_english_title_for_exact_match)
	lookup["bilby_title_en_normalized"] = lookup["title_en"].map(normalize_english_title_for_exact_match)
	return lookup.rename(
		columns={
			"original_article_url": "bilby_original_article_url",
			"site_root_url": "bilby_site_root_url",
			"original_site_root_url": "bilby_original_site_root_url",
			"published_date_cn": "bilby_published_date_cn",
			"title": "bilby_title",
			"title_en": "bilby_title_en",
			"summary": "bilby_summary",
		}
	)


def build_bilby_title_lookup(bilby_filtered: pl.DataFrame) -> pd.DataFrame:
	bilby_filtered = ensure_bilby_site_root_url(bilby_filtered)
	bilby_title_base = bilby_filtered.select(
		[
			"article_url",
			"original_article_url",
			"site_root_url",
			"original_site_root_url",
			"published_date_cn",
			"title",
			"title_en",
			"summary",
		]
	).to_pandas()

	bilby_title_lookup = pd.concat(
		[
			bilby_title_base[["article_url", "original_article_url", "site_root_url", "original_site_root_url", "published_date_cn", "title", "title_en", "summary"]]
			.assign(
				bilby_match_title=lambda df: df["title"],
				match_field="title",
			),
			bilby_title_base[["article_url", "original_article_url", "site_root_url", "original_site_root_url", "published_date_cn", "title", "title_en", "summary"]]
			.assign(
				bilby_match_title=lambda df: df["title_en"],
				match_field="title_en",
			),
		],
		ignore_index=True,
	)
	bilby_title_lookup = bilby_title_lookup.rename(
		columns={
			"original_article_url": "bilby_original_article_url",
			"site_root_url": "bilby_site_root_url",
			"original_site_root_url": "bilby_original_site_root_url",
			"published_date_cn": "bilby_published_date_cn",
			"title": "bilby_title",
			"title_en": "bilby_title_en",
			"summary": "bilby_summary",
		}
	)

	bilby_title_lookup["bilby_match_title"] = (
		bilby_title_lookup["bilby_match_title"].fillna("").astype(str).str.strip()
	)
	bilby_title_lookup["bilby_match_title_normalized"] = bilby_title_lookup.apply(
		lambda row: normalize_english_title_for_exact_match(row["bilby_match_title"])
		if row.get("match_field") == "title_en"
		else normalize_non_english_title_for_exact_match(row["bilby_match_title"]),
		axis=1,
	)
	return bilby_title_lookup.loc[
		bilby_title_lookup["bilby_match_title_normalized"].ne("")
	].drop_duplicates(subset=["bilby_match_title_normalized", "article_url", "match_field"])


def run_url_match(
	gta_filtered: pd.DataFrame,
	bilby_filtered: pl.DataFrame,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
	resolved_output_dir = ensure_output_dir(output_dir)
	bilby_url_df = build_bilby_url_lookup(bilby_filtered)
	total_bilby_docs = len(bilby_url_df)
	bilby_url_set = set(bilby_url_df["article_url_compare"].dropna().astype(str).str.strip())

	url_match_detail = gta_filtered.copy().reset_index(drop=True)
	url_match_detail["gta_row_id"] = url_match_detail.index
	url_match_detail["source_url_clean"] = (
		url_match_detail["source_url"].fillna("").astype(str).str.strip()
	)
	url_match_detail["source_url_compare"] = url_match_detail["source_url_clean"].map(normalize_compare_url)
	url_match_detail["source_title_normalized"] = url_match_detail["source_title"].map(
		normalize_title_for_exact_match
	)
	url_match_detail["matched_in_bilby_url"] = url_match_detail["source_url_compare"].isin(
		bilby_url_set
	)

	url_matched_export = url_match_detail.loc[
		url_match_detail["matched_in_bilby_url"]
	].merge(
		bilby_url_df,
		left_on="source_url_compare",
		right_on="article_url_compare",
		how="left",
	)
	url_matched_export["matched_title"] = (
		url_matched_export.get("bilby_title", pd.Series(index=url_matched_export.index, dtype="object"))
		.fillna(url_matched_export.get("title", pd.Series(index=url_matched_export.index, dtype="object")))
	)
	gta_unmatched_export = url_match_detail.loc[
		~url_match_detail["matched_in_bilby_url"]
	].copy()
	matched_article_urls = set(url_matched_export["article_url"].dropna().astype(str).tolist())
	bilby_unmatched_export = bilby_url_df.loc[
		~bilby_url_df["article_url"].astype(str).isin(matched_article_urls)
	].copy()

	output_path = resolved_output_dir / "url_match_records.xlsx"
	parquet_paths = export_match_workbook(
		url_matched_export,
		gta_unmatched_export,
		bilby_unmatched_export,
		output_path,
	)

	matched_count = int(url_match_detail["matched_in_bilby_url"].sum())
	total_count = len(url_match_detail)
	unique_matched_bilby_count = total_bilby_docs - len(bilby_unmatched_export)
	unmatched_gta_count = total_count - matched_count
	unmatched_bilby_count = len(bilby_unmatched_export)

	print("\n=== URL Match ===")
	print(f"GTA total rows: {total_count}")
	print(f"Matched by Bilby URL: {matched_count}")
	print(f"Unmatched GTA rows: {unmatched_gta_count}")
	print(f"GTA coverage: {safe_accuracy(matched_count, total_count):.2%}")
	print(f"Unique matched Bilby docs: {unique_matched_bilby_count}")
	print(f"Unmatched Bilby docs: {unmatched_bilby_count}")
	print(f"Bilby coverage: {safe_accuracy(unique_matched_bilby_count, total_bilby_docs):.2%}")
	print(f"Workbook exported to: {output_path}")
	print(f"Parquet exported to: {parquet_paths['matched']}")
	print(f"Parquet exported to: {parquet_paths['gta_unmatched']}")
	print(f"Parquet exported to: {parquet_paths['bilby_unmatched']}")

	return url_matched_export


def run_title_match(
	gta_filtered: pd.DataFrame,
	bilby_filtered: pl.DataFrame,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
	resolved_output_dir = ensure_output_dir(output_dir)
	bilby_title_lookup = build_bilby_title_lookup(bilby_filtered)
	bilby_docs = build_bilby_url_lookup(bilby_filtered)
	total_bilby_docs = len(bilby_docs)

	title_base = gta_filtered.copy().reset_index(drop=True)
	title_base["gta_row_id"] = title_base.index
	title_base["gta_title"] = title_base["source_title"].map(clean_text)
	title_base["title_lang"] = title_base["gta_title"].map(classify_title_lang)
	title_base["gta_title_normalized"] = title_base.apply(
		lambda row: normalize_english_title_for_exact_match(row["gta_title"])
		if row.get("title_lang") == "en"
		else normalize_non_english_title_for_exact_match(row["gta_title"]),
		axis=1,
	)

	english_title_base = title_base.loc[
		title_base["title_lang"].eq("en") & title_base["gta_title_normalized"].ne("")
	].copy()
	zh_title_base = title_base.loc[
		~title_base["title_lang"].eq("en") & title_base["gta_title_normalized"].ne("")
	].copy()
	english_lookup = bilby_title_lookup.loc[bilby_title_lookup["match_field"].eq("title_en")].copy()
	zh_lookup = bilby_title_lookup.loc[bilby_title_lookup["match_field"].eq("title")].copy()

	title_match_detail = pd.concat(
		[
			english_title_base.merge(
				english_lookup,
				left_on="gta_title_normalized",
				right_on="bilby_match_title_normalized",
				how="left",
			),
			zh_title_base.merge(
				zh_lookup,
				left_on="gta_title_normalized",
				right_on="bilby_match_title_normalized",
				how="left",
			),
		],
		ignore_index=True,
	).sort_values("gta_row_id").reset_index(drop=True)
	title_match_detail["date_diff_days"] = title_match_detail.apply(
		lambda row: compute_date_diff_days(row.get("published_date"), row.get("bilby_published_date_cn")),
		axis=1,
	)
	title_match_detail["date_within_one_month"] = title_match_detail.apply(
		lambda row: is_within_one_month(row.get("published_date"), row.get("bilby_published_date_cn")),
		axis=1,
	)
	title_match_detail = title_match_detail.loc[
		title_match_detail["bilby_match_title"].isna()
		| title_match_detail["date_within_one_month"]
	].copy()

	matched_row_ids = set(
		title_match_detail.loc[
			title_match_detail["bilby_match_title"].notna(), "gta_row_id"
		].tolist()
	)
	title_base["matched_in_bilby_title_or_title_en"] = title_base["gta_row_id"].isin(
		matched_row_ids
	)

	title_matched_export = title_match_detail.loc[
		title_match_detail["bilby_match_title"].notna()
	].copy()
	title_matched_export["matched_title"] = title_matched_export.get(
		"gta_title",
		pd.Series(index=title_matched_export.index, dtype="object"),
	)
	title_matched_export = title_matched_export.drop(
		columns=["bilby_match_title", "bilby_match_title_normalized"],
		errors="ignore",
	)
	gta_unmatched_export = title_base.loc[
		~title_base["matched_in_bilby_title_or_title_en"]
	].copy()
	matched_article_urls = set(title_matched_export["article_url"].dropna().astype(str).tolist())
	bilby_unmatched_export = bilby_docs.loc[
		~bilby_docs["article_url"].astype(str).isin(matched_article_urls)
	].copy()

	output_path = resolved_output_dir / "title_match_records.xlsx"
	parquet_paths = export_match_workbook(
		title_matched_export,
		gta_unmatched_export,
		bilby_unmatched_export,
		output_path,
	)

	matched_count = int(title_base["matched_in_bilby_title_or_title_en"].sum())
	total_count = len(title_base)
	unique_matched_bilby_count = total_bilby_docs - len(bilby_unmatched_export)
	unmatched_gta_count = total_count - matched_count
	unmatched_bilby_count = len(bilby_unmatched_export)

	print("\n=== Title Match ===")
	print(f"GTA total rows: {total_count}")
	print(f"Matched by Bilby title or title_en: {matched_count}")
	print(f"Unmatched GTA rows: {unmatched_gta_count}")
	print(f"GTA coverage: {safe_accuracy(matched_count, total_count):.2%}")
	print(f"Unique matched Bilby docs: {unique_matched_bilby_count}")
	print(f"Unmatched Bilby docs: {unmatched_bilby_count}")
	print(f"Bilby coverage: {safe_accuracy(unique_matched_bilby_count, total_bilby_docs):.2%}")
	print(f"Workbook exported to: {output_path}")
	print(f"Parquet exported to: {parquet_paths['matched']}")
	print(f"Parquet exported to: {parquet_paths['gta_unmatched']}")
	print(f"Parquet exported to: {parquet_paths['bilby_unmatched']}")

	return title_matched_export


def run_exact_match(
	gta_filtered: pd.DataFrame,
	bilby_filtered: pl.DataFrame,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
	url_matches = run_url_match(gta_filtered, bilby_filtered, output_dir=output_dir)
	title_matches = run_title_match(gta_filtered, bilby_filtered, output_dir=output_dir)
	combined_matches = run_combined_match(
		gta_filtered,
		bilby_filtered,
		url_matches,
		title_matches,
		output_dir=output_dir,
	)
	return {
		"url_matches": url_matches,
		"title_matches": title_matches,
		"combined_matches": combined_matches,
	}


def run_exact_match_from_files(
	gta_input_path: Path,
	bilby_input_path: Path,
	start_date: pd.Timestamp = DEFAULT_START_DATE,
	end_date: pd.Timestamp = DEFAULT_END_DATE,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
	resolved_gta_input = resolve_input_path(gta_input_path, [])
	resolved_bilby_input = resolve_input_path(
		bilby_input_path,
		[
			ROOT_DIR / "Data" / "Bilby_data_fixed" / "Bilby_2025_2026_combined_cleaned.parquet",
			ROOT_DIR / "Data" / "Bilby_data_fixed" / "Bilby_2025_2026_combined_filtered.parquet",
			ROOT_DIR / "Data" / "Bilby_data_fixed" / "Bilby_2025_2026_combined.parquet",
			ROOT_DIR / "Bilby_data" / "Bilby_2025_2026_filtered.parquet",
			ROOT_DIR / "Bilby" / "Bilby_2025_2026_filtered.parquet",
			ROOT_DIR / "Bilby_2025_2026_filtered.parquet",
		],
	)
	gta_full = normalize_gta_dates(pd.read_parquet(resolved_gta_input))
	bilby_full = normalize_bilby_dates(pl.read_parquet(resolved_bilby_input))

	summarize_dataset_dates(gta_full, bilby_full)
	selected_start_date, selected_end_date = resolve_match_date_range(start_date, end_date)

	gta_in_range_mask = gta_full["__published_date_ts"].ge(selected_start_date) & gta_full[
		"__published_date_ts"
	].lt(selected_end_date)
	gta_in_range = gta_full.loc[gta_in_range_mask].copy()
	gta_out_of_range = gta_full.loc[~gta_in_range_mask].copy()

	bilby_in_range_mask = (
		pl.col("__published_date_cn_date").is_not_null()
		& pl.col("__published_date_cn_date").ge(selected_start_date.date())
		& pl.col("__published_date_cn_date").lt(selected_end_date.date())
	)
	bilby_in_range = bilby_full.filter(bilby_in_range_mask)
	bilby_out_of_range = bilby_full.filter(~bilby_in_range_mask)

	print("\n=== Selected Date Range ===")
	print(f"Start date (inclusive): {selected_start_date.date()}")
	print(f"End date (exclusive): {selected_end_date.date()}")
	print(f"GTA in time range rows: {len(gta_in_range)}")
	print(f"Bilby in time range rows: {bilby_in_range.height}")

	export_out_of_range_data(gta_out_of_range, bilby_out_of_range, output_dir)

	gta_for_match = gta_in_range.drop(columns=["__published_date_ts"], errors="ignore")
	bilby_for_match = bilby_in_range.drop("__published_date_cn_date", strict=False)
	return run_exact_match(gta_for_match, bilby_for_match, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run exact URL and title matching on GTA and Bilby data filtered by published date."
	)
	parser.add_argument(
		"--gta-input",
		type=Path,
		default=DEFAULT_GTA_INPUT,
		help="Filtered GTA parquet file.",
	)
	parser.add_argument(
		"--bilby-input",
		type=Path,
		default=DEFAULT_BILBY_INPUT,
		help="Filtered Bilby parquet file.",
	)
	parser.add_argument(
		"--start-date",
		type=str,
		default=str(DEFAULT_START_DATE.date()),
		help="Inclusive start date in YYYY-MM-DD format.",
	)
	parser.add_argument(
		"--end-date",
		type=str,
		default=str(DEFAULT_END_DATE.date()),
		help="Exclusive end date in YYYY-MM-DD format.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory for exact match exports.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_exact_match_from_files(
		gta_input_path=args.gta_input,
		bilby_input_path=args.bilby_input,
		start_date=pd.Timestamp(args.start_date),
		end_date=pd.Timestamp(args.end_date),
		output_dir=args.output_dir,
	)


if __name__ == "__main__":
	main()
