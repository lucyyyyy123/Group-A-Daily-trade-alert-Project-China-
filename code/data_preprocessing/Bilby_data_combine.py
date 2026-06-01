from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import polars as pl


REQUIRED_COLUMNS = [

	"uuid",
	"branch_id",
	"published_at",
	"original_article_url",
	"article_url_is_proxy",
	"article_url",
	"site_root_url",
	"title",
	"subhead",
	"summary",
	"title_en",
	"translated_summary",
	"subhead_en",
	"news_line",
	"newspaper",
	"author"
]

ROOT_DIR = Path(__file__).resolve().parents[2]


def normalize_url(url: str | None) -> str | None:
	# Normalize empty/relative proxy-style URLs into a consistent canonical form.
	if url is None:
		return None

	clean_url = url.strip()
	if not clean_url:
		return None
	if clean_url.startswith("//"):
		clean_url = f"https:{clean_url}"

	return clean_url


def is_scrapeops_proxy_url(url: str | None) -> bool:
	# Detect whether this URL points to ScrapeOps proxy rather than the source site.
	clean_url = normalize_url(url)
	if clean_url is None:
		return False

	parsed_url = urlparse(clean_url)
	return parsed_url.netloc.lower() == "proxy.scrapeops.io"


def resolve_article_url(url: str | None) -> str | None:
	# If this is a proxy URL, recover the original target URL from the query string.
	clean_url = normalize_url(url)
	if clean_url is None:
		return None

	parsed_url = urlparse(clean_url)
	if parsed_url.netloc.lower() != "proxy.scrapeops.io":
		return clean_url

	target_url = parse_qs(parsed_url.query).get("url", [None])[0]
	resolved_url = normalize_url(unquote(target_url) if target_url else None)
	# Fall back to the proxy URL when extraction fails, so records are still traceable.
	return resolved_url or clean_url


def resolve_site_root_url(url: str | None) -> str | None:
	# Keep only scheme + host as a site-level key for source/domain analysis.
	clean_url = normalize_url(url)
	if not clean_url:
		return None

	parsed_url = urlparse(clean_url)
	if not parsed_url.scheme or not parsed_url.netloc:
		return None

	return urlunparse((parsed_url.scheme, parsed_url.netloc, "/", "", "", ""))


def find_input_files(bilby_dir: Path) -> list[str]:
	# Collect all monthly parquet files under Bilby_data/<year>/<month>/.
	parquet_files: list[str] = []
	for year in ("2025", "2026"):
		year_dir = bilby_dir / year
		if not year_dir.exists():
			continue
		parquet_files.extend(
			str(path)
			for path in sorted(year_dir.glob("*/*.parquet"))
			if path.is_file()
		)
	return parquet_files


def combine_bilby_parquet_files(
	bilby_dir: Path,
	output_path: Path,
) -> None:
	# Discover source files across both years before running the merge pipeline.
	input_files = find_input_files(bilby_dir)
	if not input_files:
		raise FileNotFoundError(
			f"No parquet files found under {bilby_dir / '2025'} or {bilby_dir / '2026'}."
		)

	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Read all parquet files, align missing columns, and keep only required fields.
	df = (
		pl.scan_parquet(
			input_files,
			glob=False,
			missing_columns="insert",
			cast_options=pl.ScanCastOptions(integer_cast="allow-float"),
		)
		.select(
			"uuid",
			pl.col("branch_id").cast(pl.Int64, strict=False).alias("branch_id"),
			"published_at",
			"article_url",
			"title",
			"subhead",
			"summary",
			"title_en",
			"translated_summary",
			"subhead_en",
			"news_line",
			"newspaper",
			"author",
		)
		.collect()
	)
	# Preserve the raw URL, flag proxy links, and replace proxy URLs with source URLs.
	df = df.with_columns(
		pl.col("article_url").alias("original_article_url"),
		pl.col("article_url")
		.map_elements(is_scrapeops_proxy_url, return_dtype=pl.Boolean)
		.alias("article_url_is_proxy"),
		pl.col("article_url")
		.map_elements(resolve_article_url, return_dtype=pl.String)
		.alias("article_url"),
	)
	# Add a site root column for domain-level grouping and diagnostics.
	df = df.with_columns(
		pl.col("article_url")
		.map_elements(resolve_site_root_url, return_dtype=pl.String)
		.alias("site_root_url")
	)
	# Reorder columns to the fixed output schema expected by downstream tasks.
	df = df.select(REQUIRED_COLUMNS)
	rows_before_dedup = df.height
	# Deduplicate by resolved article URL and keep the first observed record order.
	df = df.unique(subset=["article_url"], keep="first", maintain_order=True)
	# Persist the merged dataset as compressed parquet for efficient reuse.
	df.write_parquet(output_path, compression="zstd")

	print(f"Merged {len(input_files)} parquet files.")
	print(f"Rows before URL deduplication: {rows_before_dedup}")
	print(f"Duplicate article_url rows removed: {rows_before_dedup - df.height}")
	print(f"Rows written: {df.height}")
	print(f"Output: {output_path}")


def parse_args() -> argparse.Namespace:
	# Build CLI options with project-level defaults for input and output locations.
	default_bilby_dir = ROOT_DIR / "Data" / "Bilby_data_raw"
	default_output = ROOT_DIR / "Data" / "Bilby_data_fixed" / "Bilby_2025_2026_combined.parquet"

	parser = argparse.ArgumentParser(
		description="Combine all 2025 and 2026 Bilby parquet files into one parquet file."
	)
	parser.add_argument(
		"--bilby-dir",
		type=Path,
		default=default_bilby_dir,
		help="Path to the raw Bilby folder containing the 2025 and 2026 subfolders.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=default_output,
		help="Path of the combined parquet file to create.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	combine_bilby_parquet_files(
		args.bilby_dir.resolve(),
		args.output.resolve(),
	)


if __name__ == "__main__":
	main()
