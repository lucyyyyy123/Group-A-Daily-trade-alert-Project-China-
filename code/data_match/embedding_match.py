from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GTA_INPUT = ROOT_DIR / "Data" / "China_GTA_Source" / "interventions_sources_cleaned.parquet"
DEFAULT_BILBY_INPUT = ROOT_DIR / "Data" / "match_outputs" / "combined_match_bilby_unmatched.parquet"
DEFAULT_COMBINED_MATCH_INPUT = ROOT_DIR / "Data" / "match_outputs" / "combined_match_matched.parquet"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Data" / "match_outputs"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "embedding_cache"
DEFAULT_LOCAL_MODEL_PATH = ROOT_DIR / "models" / "bge-m3"
DEFAULT_START_DATE = pd.Timestamp("2025-06-01")
DEFAULT_END_DATE = pd.Timestamp("2026-03-01")
DEFAULT_BATCH_SIZE = 8
DEFAULT_SIMILARITY_CHUNK_SIZE = 1024
DEFAULT_DEVICE = "cuda"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_MAX_CANDIDATES_PER_QUERY = 1000
DEFAULT_SIMILARITY_THRESHOLD = 0.61
LOW_WEIGHT_TERM_SCORE = 0.2

LOW_WEIGHT_TERMS = {
	"about",
	"announce",
	"announces",
	"accessed on",
	"china",
	"item",
	"items",
	"notice",
	"published",
	"关于",
	"印发",
	"通知",
	"发布",
	"建议",
	"政策",
	"举措",
	"答记者问",
	"决定",
	"公告",
	"实施",
	"采取",
	"方案",
	"记者会",
	"若干",
	"应用",
	"意见",
	"商务部",
	"调整",
	"财政部",
	"中国人民银行",
	"国务院",
	"国开行",
	"中共中央",
	"制定",
	"办公厅",
	"中华人民共和国",
	"中华人民共和国国务院",
	"发言人",
	"新闻",
	"公布",
}

LOW_WEIGHT_DATE_PATTERNS = [
	re.compile(r"\b\d{4}[\-/]\d{1,2}[\-/]\d{1,2}\b", flags=re.IGNORECASE),
	re.compile(r"\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b", flags=re.IGNORECASE),
	re.compile(r"\b\d{4}\s+\d{1,2}\s+\d{1,2}\b", flags=re.IGNORECASE),
	re.compile(
		r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4}\b",
		flags=re.IGNORECASE,
	),
	re.compile(
		r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b",
		flags=re.IGNORECASE,
	),
	re.compile(r"\d{4}年\d{1,2}月(?:\d{1,2}日?)?", flags=re.IGNORECASE),
	re.compile(r"\d{1,2}月\d{1,2}日", flags=re.IGNORECASE),
]

INTERNAL_MATCH_COLUMNS = [
	"match_type",
	"gta_row_id",
	"bilby_candidate_id",
	"gta_title",
	"main_title",
	"matched_title",
	"source_url",
	"site_root_url",
	"published_date",
	"summary",
	"article_url",
	"bilby_site_root_url",
	"bilby_title",
	"bilby_title_en",
	"bilby_published_date",
	"bilby_summary",
	"embedding_similarity",
	"matched_bilby_match_field",
]

MATCHED_SHEET_COLUMNS = [
	"match_type",
	"gta_title",
	"main_title",
	"matched_title",
	"gta_title_normalized",
	"source_url",
	"site_root_url",
	"published_date",
	"summary",
	"article_url",
	"bilby_site_root_url",
	"bilby_title",
	"bilby_title_en",
	"bilby_title_normalized",
	"bilby_published_date",
	"bilby_summary",
	"embedding_similarity",
	"matched_bilby_match_field",
]

GTA_UNMATCHED_SHEET_COLUMNS = [
	"gta_title",
	"main_title",
	"gta_title_normalized",
	"source_url",
	"site_root_url",
	"published_date",
	"summary",
]

BILBY_UNMATCHED_SHEET_COLUMNS = [
	"article_url",
	"bilby_site_root_url",
	"bilby_title",
	"bilby_title_en",
	"bilby_title_normalized",
	"bilby_published_date",
	"bilby_summary",
	"embedding_similarity",
	"matched_bilby_match_field",
]

def clean_text(value: object) -> str:
	if pd.isna(value):
		return ""
	return str(value).strip()


def normalize_for_matching(value: object) -> str:
	text = clean_text(value).casefold()
	if not text:
		return ""
	normalized = "".join(character if character.isalnum() else " " for character in text)
	return re.sub(r"\s+", " ", normalized).strip()


def normalize_for_prefilter(text: str) -> str:
	return normalize_for_matching(text)


def strip_low_weight_terms(normalized_text: str) -> str:
	stripped_text = normalized_text
	for term in sorted(LOW_WEIGHT_TERMS, key=len, reverse=True):
		if not term:
			continue
		stripped_text = stripped_text.replace(term, " ")
	for date_pattern in LOW_WEIGHT_DATE_PATTERNS:
		stripped_text = date_pattern.sub(" ", stripped_text)
	return re.sub(r"\s+", " ", stripped_text).strip()


def extract_low_weight_term_hits(normalized_text: str) -> set[str]:
	hits = {term for term in LOW_WEIGHT_TERMS if term and term in normalized_text}
	if any(date_pattern.search(normalized_text) for date_pattern in LOW_WEIGHT_DATE_PATTERNS):
		hits.add("__date_pattern__")
	return hits


def normalize_title_for_output(value: object) -> str:
	return normalize_for_matching(value)


def combine_title_parts(*parts: object) -> str:
	joined = " ".join(clean_text(part) for part in parts if clean_text(part))
	return re.sub(r"\s+", " ", joined).strip()


def build_prefilter_units_from_normalized(normalized: str) -> set[str]:
	if not normalized:
		return set()
	word_units = set(re.findall(r"[a-z0-9]+", normalized))
	compact = re.sub(r"\s+", "", normalized)
	characters = [char for char in compact if char.isalnum()]
	if len(characters) < 2:
		return word_units | set(characters)
	bigrams = {"".join(characters[index : index + 2]) for index in range(len(characters) - 1)}
	return word_units | set(characters) | bigrams


def build_prefilter_profile(title: str) -> dict[str, object]:
	normalized = normalize_for_prefilter(title)
	core = strip_low_weight_terms(normalized)
	return {
		"normalized_len": len(normalized),
		"core_units": build_prefilter_units_from_normalized(core),
		"low_weight_terms": extract_low_weight_term_hits(normalized),
	}


def prefilter_score_from_profiles(
	query_profile: dict[str, object],
	candidate_profile: dict[str, object],
) -> tuple[float, int]:
	query_core_units = query_profile["core_units"]
	candidate_core_units = candidate_profile["core_units"]
	query_low_weight_terms = query_profile["low_weight_terms"]
	candidate_low_weight_terms = candidate_profile["low_weight_terms"]
	shared_core_units = len(query_core_units & candidate_core_units)
	shared_low_terms = len(query_low_weight_terms & candidate_low_weight_terms)
	weighted_shared_units = shared_core_units + LOW_WEIGHT_TERM_SCORE * shared_low_terms
	length_gap = abs(query_profile["normalized_len"] - candidate_profile["normalized_len"])
	return weighted_shared_units, -length_gap


def prefilter_score(query_title: str, candidate_title: str) -> tuple[float, int]:
	return prefilter_score_from_profiles(
		build_prefilter_profile(query_title),
		build_prefilter_profile(candidate_title),
	)


def reduce_candidates_for_query_window(
	query_subset: pd.DataFrame,
	candidate_subset: pd.DataFrame,
	max_candidates_per_query: int,
) -> tuple[pd.DataFrame, int]:
	if max_candidates_per_query <= 0 or candidate_subset.empty:
		return candidate_subset, len(candidate_subset)

	if len(candidate_subset) <= max_candidates_per_query:
		return candidate_subset, len(candidate_subset)

	candidate_titles = candidate_subset["bilby_match_title"].tolist()
	candidate_profiles = [build_prefilter_profile(title) for title in candidate_titles]
	selected_indices: set[int] = set()

	for gta_title in query_subset["gta_query_title"].tolist():
		query_profile = build_prefilter_profile(gta_title)
		scored_candidates: list[tuple[float, int, int]] = []
		for candidate_index, candidate_profile in enumerate(candidate_profiles):
			shared_units, negative_length_gap = prefilter_score_from_profiles(
				query_profile,
				candidate_profile,
			)
			if shared_units == 0:
				continue
			scored_candidates.append((shared_units, negative_length_gap, candidate_index))

		if scored_candidates:
			scored_candidates.sort(reverse=True)
			selected_indices.update(
				candidate_index
				for _, _, candidate_index in scored_candidates[:max_candidates_per_query]
			)
		else:
			query_length = query_profile["normalized_len"]
			fallback_order = sorted(
				range(len(candidate_profiles)),
				key=lambda candidate_index: abs(
					query_length - candidate_profiles[candidate_index]["normalized_len"]
				),
			)
			selected_indices.update(fallback_order[:max_candidates_per_query])

	if not selected_indices:
		return candidate_subset.head(max_candidates_per_query).reset_index(drop=True), len(candidate_subset)

	reduced_subset = candidate_subset.iloc[sorted(selected_indices)].reset_index(drop=True)
	return reduced_subset, len(candidate_subset)


def ensure_directory(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path.resolve()


def resolve_output_bundle_paths(output_dir: Path) -> tuple[Path, Path]:
	base_workbook_stem = "embedding_match_matched_records"
	base_checkpoint_stem = "embedding_match_matched_records_checkpoint"
	for suffix_index in range(0, 10000):
		suffix = "" if suffix_index == 0 else f"({suffix_index})"
		output_path = output_dir / f"{base_workbook_stem}{suffix}.xlsx"
		checkpoint_path = output_dir / f"{base_checkpoint_stem}{suffix}.parquet"
		base_name = f"{base_workbook_stem}{suffix}"
		parquet_paths = [
			output_dir / f"{base_name}_matched.parquet",
			output_dir / f"{base_name}_gta_unmatched.parquet",
			output_dir / f"{base_name}_bilby_unmatched.parquet",
		]
		if output_path.exists() or checkpoint_path.exists() or any(path.exists() for path in parquet_paths):
			continue
		return output_path, checkpoint_path
	raise RuntimeError("Could not resolve a collision-free output filename after 10000 attempts.")


def format_seconds(seconds: float) -> str:
	if seconds < 60:
		return f"{seconds:.1f}s"
	minutes, remaining_seconds = divmod(seconds, 60)
	if minutes < 60:
		return f"{int(minutes)}m {remaining_seconds:.1f}s"
	hours, remaining_minutes = divmod(minutes, 60)
	return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def load_filtered_data(
	gta_input_path: Path,
	bilby_input_path: Path,
	start_date: pd.Timestamp,
	end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	# Step 1: Load GTA and Bilby source tables.
	gta_filtered = pd.read_parquet(gta_input_path.resolve())
	bilby_filtered = pd.read_parquet(bilby_input_path.resolve())

	if "published_date" in gta_filtered.columns:
		gta_filtered["published_date"] = pd.to_datetime(gta_filtered["published_date"], errors="coerce")

	bilby_date_col = next(
		(column for column in ["bilby_published_date", "published_date_cn", "published_date"] if column in bilby_filtered.columns),
		None,
	)
	if bilby_date_col is not None:
		bilby_filtered[bilby_date_col] = pd.to_datetime(bilby_filtered[bilby_date_col], errors="coerce")

		# Step 2: First narrow GTA rows by Bilby unmatched date span.
		if "published_date" in gta_filtered.columns:
			bilby_valid_dates = bilby_filtered[bilby_date_col].dropna()
			if not bilby_valid_dates.empty:
				bilby_min_date = bilby_valid_dates.min()
				bilby_max_date = bilby_valid_dates.max()
				gta_filtered = gta_filtered.loc[
					gta_filtered["published_date"].between(bilby_min_date, bilby_max_date, inclusive="both")
				].copy()

	# Step 3: Apply configured global date window on both datasets.
	if "published_date" in gta_filtered.columns:
		gta_filtered = gta_filtered.loc[
			gta_filtered["published_date"].ge(start_date) & gta_filtered["published_date"].lt(end_date)
		].copy()

	if bilby_date_col is not None:
		bilby_filtered = bilby_filtered.loc[
			bilby_filtered[bilby_date_col].ge(start_date) & bilby_filtered[bilby_date_col].lt(end_date)
		].copy()

	return gta_filtered, bilby_filtered


def build_gta_query_frame(gta_filtered: pd.DataFrame) -> pd.DataFrame:
	query_df = gta_filtered.copy().reset_index(drop=True)
	query_df["gta_row_id"] = query_df.index
	title_source_column = "gta_title" if "gta_title" in query_df.columns else "source_title"
	query_df["gta_title"] = query_df[title_source_column].map(clean_text)
	query_df["main_title"] = query_df.get("main_title", "").map(clean_text)
	query_df["gta_query_title"] = query_df.apply(
		lambda row: combine_title_parts(row.get("main_title"), row.get("gta_title")),
		axis=1,
	)
	for column, default_value in {
		"source_url": "",
		"site_root_url": "",
		"summary": "",
	}.items():
		if column not in query_df.columns:
			query_df[column] = default_value
	query_df = query_df.loc[query_df["gta_query_title"].ne("")].copy()
	query_df["published_date"] = pd.to_datetime(query_df["published_date"], errors="coerce")
	query_df = query_df.loc[query_df["published_date"].notna()].copy()
	query_df = query_df.reset_index(drop=True)
	query_df["query_embedding_index"] = query_df.index
	query_df["published_date_key"] = query_df["published_date"].dt.normalize()
	return query_df


def build_bilby_candidate_frame(bilby_filtered: pd.DataFrame) -> pd.DataFrame:
	date_column = next(
		(column for column in ["bilby_published_date", "published_date_cn", "published_date"] if column in bilby_filtered.columns),
		None,
	)
	if date_column is None:
		raise ValueError("Bilby unmatched input must contain bilby_published_date or published_date_cn")

	bilby_title_column = "bilby_title" if "bilby_title" in bilby_filtered.columns else "title"
	bilby_title_en_column = "bilby_title_en" if "bilby_title_en" in bilby_filtered.columns else "title_en"
	article_url_column = "article_url" if "article_url" in bilby_filtered.columns else "matched_bilby_article_url"
	site_root_column = (
		"bilby_site_root_url"
		if "bilby_site_root_url" in bilby_filtered.columns
		else ("site_root_url" if "site_root_url" in bilby_filtered.columns else None)
	)

	base_columns = [date_column, article_url_column, bilby_title_column, bilby_title_en_column]
	if "bilby_summary" in bilby_filtered.columns:
		bilby_summary_column = "bilby_summary"
	elif "summary" in bilby_filtered.columns:
		bilby_summary_column = "summary"
	else:
		bilby_summary_column = None
	if bilby_summary_column is not None:
		base_columns.append(bilby_summary_column)
	if site_root_column is not None:
		base_columns.append(site_root_column)
	bilby_title_base = bilby_filtered[base_columns].copy().rename(
		columns={
			date_column: "published_date_cn",
			article_url_column: "article_url",
			bilby_title_column: "bilby_title",
			bilby_title_en_column: "bilby_title_en",
		}
	)
	if bilby_summary_column is not None:
		bilby_title_base = bilby_title_base.rename(columns={bilby_summary_column: "bilby_summary"})
	else:
		bilby_title_base["bilby_summary"] = ""
	if site_root_column is not None:
		bilby_title_base = bilby_title_base.rename(columns={site_root_column: "bilby_site_root_url"})
	else:
		bilby_title_base["bilby_site_root_url"] = ""
	bilby_title_base["bilby_title"] = bilby_title_base["bilby_title"].map(clean_text)
	bilby_title_base["bilby_title_en"] = bilby_title_base["bilby_title_en"].map(clean_text)
	bilby_title_base["bilby_summary"] = bilby_title_base["bilby_summary"].map(clean_text)
	bilby_title_base["bilby_site_root_url"] = bilby_title_base["bilby_site_root_url"].map(clean_text)
	bilby_title_base["published_date_cn"] = pd.to_datetime(
		bilby_title_base["published_date_cn"],
		errors="coerce",
	).dt.normalize()
	bilby_title_base = bilby_title_base.loc[bilby_title_base["published_date_cn"].notna()].copy()

	candidate_df = (
		bilby_title_base.loc[
			bilby_title_base["bilby_title"].ne("") | bilby_title_base["bilby_title_en"].ne(""),
			[
				"published_date_cn",
				"article_url",
				"bilby_site_root_url",
				"bilby_title",
				"bilby_title_en",
				"bilby_summary",
			],
		]
		.copy()
	)
	candidate_df["bilby_match_field"] = "bilby_title_combined"
	candidate_df["bilby_match_title"] = candidate_df.apply(
		lambda row: combine_title_parts(row.get("bilby_title"), row.get("bilby_title_en")),
		axis=1,
	)
	candidate_df["bilby_match_title"] = candidate_df["bilby_match_title"].map(normalize_for_matching)
	candidate_df = candidate_df.loc[candidate_df["bilby_match_title"].ne("")].copy()
	candidate_df = candidate_df.drop_duplicates(
		subset=[
			"published_date_cn",
			"bilby_match_title",
			"bilby_match_field",
			"article_url",
		]
	).reset_index(drop=True)
	candidate_df["bilby_candidate_id"] = candidate_df.index
	return candidate_df


def build_cache_key(candidate_subset: pd.DataFrame) -> str:
	hasher = hashlib.sha1()
	for row in candidate_subset[["article_url", "bilby_match_field", "bilby_match_title"]].itertuples(index=False):
		hasher.update("||".join(clean_text(value) for value in row).encode("utf-8"))
		hasher.update(b"\n")
	return hasher.hexdigest()[:16]


def load_sentence_transformer(device: str) -> object:
	try:
		from sentence_transformers import SentenceTransformer
	except Exception as exc:
		raise RuntimeError(f"Embedding dependency error: {type(exc).__name__}: {exc}") from exc

	def _load_model(target_device: str) -> object:
		model_path = DEFAULT_LOCAL_MODEL_PATH.resolve()
		if model_path.exists():
			print(f"Loading multilingual embedding model from local path: {model_path}")
			return SentenceTransformer(str(model_path), device=target_device, local_files_only=True)

		model_name_or_path = "BAAI/bge-m3"
		print(
			f"Local embedding model not found at {model_path}. "
			f"Falling back to remote model: {model_name_or_path}"
		)
		return SentenceTransformer(model_name_or_path, device=target_device)

	requested_device = clean_text(device).lower() or DEFAULT_DEVICE
	try:
		return _load_model(requested_device)
	except Exception as exc:
		error_text = clean_text(exc)
		should_fallback_to_cpu = requested_device.startswith("cuda") and (
			"cuda" in error_text.casefold()
			or "cudnn" in error_text.casefold()
			or "not compiled with cuda" in error_text.casefold()
		)
		if not should_fallback_to_cpu:
			raise
		print(
			f"Failed to initialize embedding model on device '{requested_device}': {error_text}. "
			"Falling back to CPU."
		)
		return _load_model("cpu")


def encode_texts(
	model: object,
	texts: list[str],
	batch_size: int,
	show_progress_bar: bool,
) -> np.ndarray:
	return model.encode(
		texts,
		normalize_embeddings=True,
		convert_to_numpy=True,
		batch_size=batch_size,
		show_progress_bar=show_progress_bar,
	)


def encode_query_texts(
	model: object,
	texts: list[str],
	batch_size: int,
	show_progress_bar: bool,
) -> np.ndarray:
	prefix = "Represent this title for retrieving similar titles: "
	prefixed_texts = [f"{prefix}{normalize_for_matching(text)}" for text in texts]
	return encode_texts(model, prefixed_texts, batch_size=batch_size, show_progress_bar=show_progress_bar)


def get_or_create_candidate_embeddings(
	model: object,
	candidate_subset: pd.DataFrame,
	cache_dir: Path,
	batch_size: int,
	show_progress_bar: bool,
) -> tuple[Path, bool]:
	resolved_cache_dir = ensure_directory(cache_dir)
	window_start = pd.Timestamp(candidate_subset["published_date_cn"].min()).date().isoformat()
	window_end = pd.Timestamp(candidate_subset["published_date_cn"].max()).date().isoformat()
	cache_key = build_cache_key(candidate_subset)
	cache_path = resolved_cache_dir / f"bilby_{window_start}_{window_end}_multi_{cache_key}.npy"

	if cache_path.exists():
		print(f"Using cached Bilby embeddings: {cache_path}")
		return cache_path, True

	texts = candidate_subset["bilby_match_title"].tolist()
	embeddings = encode_texts(
		model,
		texts,
		batch_size=batch_size,
		show_progress_bar=show_progress_bar,
	)
	np.save(cache_path, embeddings)
	print(f"Saved Bilby embeddings cache: {cache_path}")
	return cache_path, False


def compute_matches_above_threshold(
	query_embeddings: np.ndarray,
	candidate_embeddings: np.ndarray,
	threshold: float,
	chunk_size: int,
	) -> list[list[tuple[int, float]]]:
	query_count = query_embeddings.shape[0]
	matches_by_query: list[list[tuple[int, float]]] = [[] for _ in range(query_count)]
	if candidate_embeddings.shape[0] == 0:
		return matches_by_query

	for start in range(0, candidate_embeddings.shape[0], chunk_size):
		end = min(start + chunk_size, candidate_embeddings.shape[0])
		chunk_embeddings = candidate_embeddings[start:end]
		chunk_scores = query_embeddings @ chunk_embeddings.T
		for query_position in range(query_count):
			eligible_positions = np.flatnonzero(chunk_scores[query_position] > threshold)
			for position in eligible_positions.tolist():
				matches_by_query[query_position].append(
					(start + int(position), float(chunk_scores[query_position, position]))
				)

	for query_position in range(query_count):
		matches_by_query[query_position].sort(key=lambda item: item[1], reverse=True)

	return matches_by_query


def build_match_rows(
	query_subset: pd.DataFrame,
	candidate_subset: pd.DataFrame,
	query_matches: list[list[tuple[int, float]]],
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for query_position, gta_row in enumerate(query_subset.itertuples(index=False)):
		for candidate_index, score in query_matches[query_position]:
			candidate_row = candidate_subset.iloc[int(candidate_index)]
			article_url = clean_text(candidate_row["article_url"])
			rows.append(
				{
					"match_type": "embedding_match",
					"gta_row_id": gta_row.gta_row_id,
					"bilby_candidate_id": int(candidate_row["bilby_candidate_id"]),
					"gta_title": gta_row.gta_title,
					"main_title": gta_row.main_title,
					"matched_title": combine_title_parts(
						candidate_row["bilby_title"],
						candidate_row["bilby_title_en"],
					),
					"source_url": gta_row.source_url,
					"site_root_url": gta_row.site_root_url,
					"published_date": gta_row.published_date,
					"summary": gta_row.summary,
					"article_url": article_url,
					"bilby_site_root_url": candidate_row["bilby_site_root_url"],
					"bilby_title": candidate_row["bilby_title"],
					"bilby_title_en": candidate_row["bilby_title_en"],
					"bilby_published_date": candidate_row["published_date_cn"],
					"bilby_summary": candidate_row["bilby_summary"],
					"matched_bilby_match_field": candidate_row["bilby_match_field"],
					"embedding_similarity": score,
				}
			)
	return rows


def build_match_rows_for_query(
	query_row: pd.Series,
	candidate_subset: pd.DataFrame,
	query_matches_row: list[tuple[int, float]],
) -> list[dict[str, object]]:
	query_subset = pd.DataFrame([query_row])
	return build_match_rows(
		query_subset=query_subset,
		candidate_subset=candidate_subset,
		query_matches=[query_matches_row],
	)


def load_combined_match_records(combined_match_input_path: Path | None) -> pd.DataFrame:
	if combined_match_input_path is None:
		return pd.DataFrame(columns=INTERNAL_MATCH_COLUMNS)

	resolved_path = combined_match_input_path.resolve()
	if not resolved_path.exists():
		print(f"Combined match parquet not found, skip merge: {resolved_path}")
		return pd.DataFrame(columns=INTERNAL_MATCH_COLUMNS)

	combined_df = pd.read_parquet(resolved_path)
	if combined_df.empty:
		return pd.DataFrame(columns=INTERNAL_MATCH_COLUMNS)

	prepared = combined_df.copy()
	if "match_type" not in prepared.columns:
		prepared["match_type"] = "title_match"
	prepared["gta_title"] = prepared.get("gta_title", pd.Series("", index=prepared.index)).map(clean_text)
	prepared["main_title"] = prepared.get("main_title", pd.Series("", index=prepared.index)).map(clean_text)
	prepared["matched_title"] = prepared.get("matched_title", prepared.get("bilby_title", "")).map(clean_text)
	prepared["source_url"] = prepared.get("source_url", pd.Series("", index=prepared.index)).map(clean_text)
	prepared["site_root_url"] = prepared.get("site_root_url", pd.Series("", index=prepared.index)).map(clean_text)
	prepared["summary"] = prepared.get("summary", pd.Series("", index=prepared.index)).map(clean_text)
	prepared["article_url"] = prepared.get("article_url", pd.Series("", index=prepared.index)).map(clean_text)
	prepared["bilby_site_root_url"] = prepared.get(
		"bilby_site_root_url",
		pd.Series("", index=prepared.index),
	).map(clean_text)
	prepared["bilby_title"] = prepared.get("bilby_title", pd.Series("", index=prepared.index)).map(clean_text)
	prepared["bilby_title_en"] = prepared.get("bilby_title_en", pd.Series("", index=prepared.index)).map(clean_text)
	prepared["bilby_summary"] = prepared.get("bilby_summary", pd.Series("", index=prepared.index)).map(clean_text)
	prepared["published_date"] = pd.to_datetime(prepared.get("published_date", pd.NaT), errors="coerce")
	prepared["bilby_published_date"] = pd.to_datetime(
		prepared.get("bilby_published_date", prepared.get("bilby_published_date_cn", pd.NaT)),
		errors="coerce",
	)
	prepared["gta_row_id"] = pd.to_numeric(
		prepared.get("gta_row_id", pd.Series([pd.NA] * len(prepared), index=prepared.index)),
		errors="coerce",
	)
	prepared["bilby_candidate_id"] = pd.NA
	prepared["embedding_similarity"] = pd.NA
	prepared["matched_bilby_match_field"] = prepared.get(
		"matched_bilby_match_field",
		prepared["match_type"],
	)

	for column in INTERNAL_MATCH_COLUMNS:
		if column not in prepared.columns:
			prepared[column] = pd.NA

	prepared = prepared[INTERNAL_MATCH_COLUMNS].copy()
	prepared = prepared.loc[prepared["article_url"].ne("")].copy()
	return prepared


def save_results_checkpoint(result_rows: list[dict[str, object]], output_path: Path) -> None:
	checkpoint_df = pd.DataFrame(result_rows, columns=INTERNAL_MATCH_COLUMNS)
	if checkpoint_df.empty:
		checkpoint_df = pd.DataFrame(columns=INTERNAL_MATCH_COLUMNS)
	checkpoint_df.to_parquet(output_path, index=False)


def build_output_sheets(
	query_df: pd.DataFrame,
	candidate_df: pd.DataFrame,
	result_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if result_df.empty:
		matched_sheet = pd.DataFrame(columns=MATCHED_SHEET_COLUMNS)
	else:
		matched_sheet = pd.DataFrame(
			{
				"match_type": result_df["match_type"],
				"gta_title": result_df["gta_title"],
				"main_title": result_df["main_title"],
				"matched_title": result_df["matched_title"],
				"gta_title_normalized": result_df["gta_title"].map(normalize_title_for_output),
				"source_url": result_df["source_url"],
				"site_root_url": result_df["site_root_url"],
				"published_date": result_df["published_date"],
				"summary": result_df["summary"],
				"article_url": result_df["article_url"],
				"bilby_site_root_url": result_df["bilby_site_root_url"],
				"bilby_title": result_df["bilby_title"],
				"bilby_title_en": result_df["bilby_title_en"],
				"bilby_title_normalized": result_df["bilby_title"].map(normalize_title_for_output),
				"bilby_published_date": result_df["bilby_published_date"],
				"bilby_summary": result_df["bilby_summary"],
				"embedding_similarity": result_df["embedding_similarity"],
				"matched_bilby_match_field": result_df["matched_bilby_match_field"],
			}
		)
		matched_sheet = matched_sheet[MATCHED_SHEET_COLUMNS].copy()

	matched_gta_ids = set(result_df["gta_row_id"].tolist()) if not result_df.empty else set()
	gta_unmatched_source = query_df.loc[~query_df["gta_row_id"].isin(matched_gta_ids)].copy()
	gta_unmatched_sheet = pd.DataFrame(
		{
			"gta_title": gta_unmatched_source["gta_title"],
			"main_title": gta_unmatched_source["main_title"],
			"gta_title_normalized": gta_unmatched_source["gta_title"].map(normalize_title_for_output),
			"source_url": gta_unmatched_source["source_url"],
			"site_root_url": gta_unmatched_source["site_root_url"],
			"published_date": gta_unmatched_source["published_date"],
			"summary": gta_unmatched_source["summary"],
		}
	)
	gta_unmatched_sheet = gta_unmatched_sheet[GTA_UNMATCHED_SHEET_COLUMNS].copy()

	if "bilby_candidate_id" in result_df.columns and "bilby_candidate_id" in candidate_df.columns:
		matched_candidate_ids = set(result_df["bilby_candidate_id"].dropna().astype(int).tolist())
		bilby_unmatched_source = candidate_df.loc[
			~candidate_df["bilby_candidate_id"].isin(matched_candidate_ids)
		].copy()
	else:
		if result_df.empty:
			matched_candidate_keys: set[tuple[str, str]] = set()
		else:
			matched_candidate_keys = {
				(
					clean_text(row[0]),
					clean_text(row[1]),
				)
				for row in result_df[["article_url", "matched_bilby_match_field"]].itertuples(index=False, name=None)
			}

		candidate_keys = candidate_df[["article_url", "bilby_match_field"]].apply(
			lambda row: (
				clean_text(row["article_url"]),
				clean_text(row["bilby_match_field"]),
			),
			axis=1,
		)
		bilby_unmatched_source = candidate_df.loc[~candidate_keys.isin(matched_candidate_keys)].copy()
	bilby_unmatched_sheet = pd.DataFrame(
		{
			"article_url": bilby_unmatched_source["article_url"],
			"bilby_site_root_url": bilby_unmatched_source["bilby_site_root_url"],
			"bilby_title": bilby_unmatched_source["bilby_title"],
			"bilby_title_en": bilby_unmatched_source["bilby_title_en"],
			"bilby_title_normalized": bilby_unmatched_source["bilby_title"].map(normalize_title_for_output),
			"bilby_published_date": bilby_unmatched_source["published_date_cn"],
			"bilby_summary": bilby_unmatched_source["bilby_summary"],
			"embedding_similarity": pd.NA,
			"matched_bilby_match_field": bilby_unmatched_source["bilby_match_field"],
		}
	)
	bilby_unmatched_sheet = bilby_unmatched_sheet[BILBY_UNMATCHED_SHEET_COLUMNS].copy()

	return matched_sheet, gta_unmatched_sheet, bilby_unmatched_sheet


def save_output_workbook(
	output_path: Path,
	matched_sheet: pd.DataFrame,
	gta_unmatched_sheet: pd.DataFrame,
	bilby_unmatched_sheet: pd.DataFrame,
) -> None:
	with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
		matched_sheet.to_excel(writer, sheet_name="matched", index=False)
		gta_unmatched_sheet.to_excel(writer, sheet_name="gta_unmatched", index=False)
		bilby_unmatched_sheet.to_excel(writer, sheet_name="bilby_unmatched", index=False)


def save_output_parquets(
	output_path: Path,
	matched_sheet: pd.DataFrame,
	gta_unmatched_sheet: pd.DataFrame,
	bilby_unmatched_sheet: pd.DataFrame,
) -> dict[str, Path]:
	base_path = output_path.with_suffix("")
	parquet_paths = {
		"matched": base_path.with_name(f"{base_path.name}_matched.parquet"),
		"gta_unmatched": base_path.with_name(f"{base_path.name}_gta_unmatched.parquet"),
		"bilby_unmatched": base_path.with_name(f"{base_path.name}_bilby_unmatched.parquet"),
	}
	matched_sheet.to_parquet(parquet_paths["matched"], index=False)
	gta_unmatched_sheet.to_parquet(parquet_paths["gta_unmatched"], index=False)
	bilby_unmatched_sheet.to_parquet(parquet_paths["bilby_unmatched"], index=False)
	return parquet_paths


def empty_result(
	reason: str,
	output_path: Path,
	query_df: pd.DataFrame,
	candidate_df: pd.DataFrame,
	base_result_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
	print("\n=== Embedding Match Skipped ===")
	print(f"Current Python: {sys.executable}")
	print(reason)
	empty_df = (
		pd.DataFrame(columns=INTERNAL_MATCH_COLUMNS)
		if base_result_df is None
		else base_result_df.copy()
	)
	matched_sheet, gta_unmatched_sheet, bilby_unmatched_sheet = build_output_sheets(
		query_df=query_df,
		candidate_df=candidate_df,
		result_df=empty_df,
	)
	save_output_workbook(output_path, matched_sheet, gta_unmatched_sheet, bilby_unmatched_sheet)
	parquet_paths = save_output_parquets(
		output_path=output_path,
		matched_sheet=matched_sheet,
		gta_unmatched_sheet=gta_unmatched_sheet,
		bilby_unmatched_sheet=bilby_unmatched_sheet,
	)
	print(f"Empty embedding export written to: {output_path}")
	print(f"Parquet exported to: {parquet_paths['matched']}")
	print(f"Parquet exported to: {parquet_paths['gta_unmatched']}")
	print(f"Parquet exported to: {parquet_paths['bilby_unmatched']}")
	return empty_df


def run_embedding_match(
	gta_filtered: pd.DataFrame,
	bilby_filtered: pd.DataFrame,
	combined_match_input_path: Path | None = DEFAULT_COMBINED_MATCH_INPUT,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
	cache_dir: Path = DEFAULT_CACHE_DIR,
	device: str = DEFAULT_DEVICE,
	batch_size: int = DEFAULT_BATCH_SIZE,
	window_days: int = DEFAULT_WINDOW_DAYS,
	similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
	max_candidates_per_query: int = DEFAULT_MAX_CANDIDATES_PER_QUERY,
	similarity_chunk_size: int = DEFAULT_SIMILARITY_CHUNK_SIZE,
	show_progress_bar: bool = True,
) -> pd.DataFrame:
	# Step 1: Validate runtime options.
	if batch_size <= 0:
		raise ValueError("batch_size must be a positive integer")
	if window_days < 0:
		raise ValueError("window_days must be a non-negative integer")
	if not 0 <= similarity_threshold <= 1:
		raise ValueError("similarity_threshold must be between 0 and 1")
	if max_candidates_per_query < 0:
		raise ValueError("max_candidates_per_query must be a non-negative integer")
	if similarity_chunk_size <= 0:
		raise ValueError("similarity_chunk_size must be a positive integer")

	# Step 2: Build query/candidate frames and output/checkpoint paths.
	resolved_output_dir = ensure_directory(output_dir)
	output_path, checkpoint_path = resolve_output_bundle_paths(resolved_output_dir)
	query_df = build_gta_query_frame(gta_filtered)
	candidate_df = build_bilby_candidate_frame(bilby_filtered)
	combined_match_df = load_combined_match_records(combined_match_input_path)
	resolved_cache_dir = ensure_directory(cache_dir)
	save_results_checkpoint(combined_match_df.to_dict(orient="records"), checkpoint_path)

	if query_df.empty:
		return empty_result(
			"No GTA titles available for embedding match.",
			output_path,
			query_df=query_df,
			candidate_df=candidate_df,
			base_result_df=combined_match_df,
		)
	if candidate_df.empty:
		return empty_result(
			"No Bilby title candidates available for embedding match.",
			output_path,
			query_df=query_df,
			candidate_df=candidate_df,
			base_result_df=combined_match_df,
		)

	try:
		model = load_sentence_transformer(device)
	except Exception as exc:
		return empty_result(
			str(exc),
			output_path,
			query_df=query_df,
			candidate_df=candidate_df,
			base_result_df=combined_match_df,
		)

	result_rows: list[dict[str, object]] = []
	window_delta = pd.Timedelta(days=window_days)
	total_start_time = time.perf_counter()

	query_encode_start_time = time.perf_counter()
	all_query_embeddings = encode_query_texts(
		model,
		query_df["gta_query_title"].tolist(),
		batch_size=batch_size,
		show_progress_bar=show_progress_bar,
	)
	query_encode_elapsed = time.perf_counter() - query_encode_start_time

	candidate_embedding_start_time = time.perf_counter()
	cache_path, cache_hit = get_or_create_candidate_embeddings(
		model,
		candidate_df,
		cache_dir=resolved_cache_dir,
		batch_size=batch_size,
		show_progress_bar=show_progress_bar,
	)
	all_candidate_embeddings = np.load(cache_path, mmap_mode="r")
	candidate_embedding_elapsed = time.perf_counter() - candidate_embedding_start_time
	cache_status = "cache hit" if cache_hit else "fresh encode"

	print(
		f"Starting embedding match: GTA queries={len(query_df)}, Bilby candidates={len(candidate_df)}, preloaded exact matches={len(combined_match_df)}, threshold={similarity_threshold}, coverage_filter=disabled, window_days={window_days}, max_candidates_per_query={max_candidates_per_query}"
	)
	print(f"Full query embeddings completed in {format_seconds(query_encode_elapsed)}")
	print(
		f"Full Bilby embeddings ready in {format_seconds(candidate_embedding_elapsed)} ({cache_status})"
	)
	print("Multilingual embedding is enabled; cross-language title matches (for example Chinese to English) are allowed.")

	# Step 3: Process each GTA date bucket with prefilter + embedding similarity.
	for published_date_key in sorted(query_df["published_date_key"].dropna().unique()):
		bucket_start_time = time.perf_counter()
		query_subset = query_df.loc[
			query_df["published_date_key"].eq(published_date_key)
		].reset_index(drop=True)
		window_start = pd.Timestamp(published_date_key) - window_delta
		window_end = pd.Timestamp(published_date_key) + window_delta
		candidate_subset = candidate_df.loc[
			candidate_df["published_date_cn"].ge(window_start)
			& candidate_df["published_date_cn"].le(window_end)
		].reset_index(drop=True)

		if query_subset.empty or candidate_subset.empty:
			continue

		prefilter_start_time = time.perf_counter()
		candidate_subset, candidate_count_before_prefilter = reduce_candidates_for_query_window(
			query_subset=query_subset,
			candidate_subset=candidate_subset,
			max_candidates_per_query=max_candidates_per_query,
		)
		prefilter_elapsed = time.perf_counter() - prefilter_start_time

		print(
			f"Embedding published_date={pd.Timestamp(published_date_key).date()} window={window_start.date()}~{window_end.date()}: GTA={len(query_subset)} Bilby candidates={candidate_count_before_prefilter} -> {len(candidate_subset)} threshold>{similarity_threshold}, coverage_filter=disabled"
		)
		print(f"  Prefilter completed in {format_seconds(prefilter_elapsed)}")

		query_embedding_indices = query_subset["query_embedding_index"].to_numpy(dtype=np.int64)
		query_embeddings = all_query_embeddings[query_embedding_indices]
		candidate_embedding_indices = candidate_subset["bilby_candidate_id"].to_numpy(dtype=np.int64)
		candidate_embeddings = all_candidate_embeddings[candidate_embedding_indices]

		similarity_start_time = time.perf_counter()
		query_matches = compute_matches_above_threshold(
			query_embeddings=query_embeddings,
			candidate_embeddings=candidate_embeddings,
			threshold=similarity_threshold,
			chunk_size=similarity_chunk_size,
		)
		similarity_elapsed = time.perf_counter() - similarity_start_time
		print(f"  Similarity computation completed in {format_seconds(similarity_elapsed)}")
		bucket_rows_before = len(result_rows)
		for query_position in range(len(query_subset)):
			query_rows = build_match_rows_for_query(
				query_row=query_subset.iloc[query_position],
				candidate_subset=candidate_subset,
				query_matches_row=query_matches[query_position],
			)
			result_rows.extend(query_rows)
		save_results_checkpoint(
			combined_match_df.to_dict(orient="records") + result_rows,
			checkpoint_path,
		)
		print(
			f"  Checkpoint saved after published_date={pd.Timestamp(published_date_key).date()} with {len(result_rows)} total exported rows (+{len(result_rows) - bucket_rows_before} in this bucket)"
		)
		bucket_elapsed = time.perf_counter() - bucket_start_time
		print(f"  Bucket completed in {format_seconds(bucket_elapsed)}")

	if not result_rows and combined_match_df.empty:
		return empty_result(
			"No embedding matches were produced for the available date-window buckets.",
			output_path,
			query_df=query_df,
			candidate_df=candidate_df,
			base_result_df=combined_match_df,
		)

	# Step 4: Keep best Bilby hits and export final outputs.
	embedding_result_df = pd.DataFrame(result_rows, columns=INTERNAL_MATCH_COLUMNS)
	result_df = pd.concat([combined_match_df, embedding_result_df], ignore_index=True, sort=False)
	if result_df.empty:
		result_df = pd.DataFrame(columns=INTERNAL_MATCH_COLUMNS)
	matched_sheet, gta_unmatched_sheet, bilby_unmatched_sheet = build_output_sheets(
		query_df=query_df,
		candidate_df=candidate_df,
		result_df=result_df,
	)
	save_output_workbook(output_path, matched_sheet, gta_unmatched_sheet, bilby_unmatched_sheet)
	parquet_paths = save_output_parquets(
		output_path=output_path,
		matched_sheet=matched_sheet,
		gta_unmatched_sheet=gta_unmatched_sheet,
		bilby_unmatched_sheet=bilby_unmatched_sheet,
	)

	query_count = query_df["gta_row_id"].nunique()
	covered_count = int(result_df["gta_row_id"].nunique())
	print("\n=== Embedding Match ===")
	print(f"GTA total rows: {len(gta_filtered)}")
	print(f"GTA rows with non-empty titles: {query_count}")
	print(f"Candidate rows exported (combined exact + embedding): {len(result_df)}")
	print(f"Combined exact rows included: {len(combined_match_df)}")
	print(f"Embedding rows added: {len(embedding_result_df)}")
	print(f"GTA rows with at least one match: {covered_count}/{query_count}")
	print(f"GTA unmatched rows: {len(gta_unmatched_sheet)}")
	print(f"Bilby unmatched rows: {len(bilby_unmatched_sheet)}")
	print(f"Total runtime: {format_seconds(time.perf_counter() - total_start_time)}")
	print(f"Embedding records exported to: {output_path}")
	print(f"Parquet exported to: {parquet_paths['matched']}")
	print(f"Parquet exported to: {parquet_paths['gta_unmatched']}")
	print(f"Parquet exported to: {parquet_paths['bilby_unmatched']}")
	return result_df


def run_embedding_match_from_files(
	gta_input_path: Path,
	bilby_input_path: Path,
	combined_match_input_path: Path | None = DEFAULT_COMBINED_MATCH_INPUT,
	start_date: pd.Timestamp = DEFAULT_START_DATE,
	end_date: pd.Timestamp = DEFAULT_END_DATE,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
	cache_dir: Path = DEFAULT_CACHE_DIR,
	device: str = DEFAULT_DEVICE,
	batch_size: int = DEFAULT_BATCH_SIZE,
	window_days: int = DEFAULT_WINDOW_DAYS,
	similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
	max_candidates_per_query: int = DEFAULT_MAX_CANDIDATES_PER_QUERY,
	similarity_chunk_size: int = DEFAULT_SIMILARITY_CHUNK_SIZE,
	show_progress_bar: bool = True,
) -> pd.DataFrame:
	gta_filtered, bilby_filtered = load_filtered_data(
		gta_input_path=gta_input_path,
		bilby_input_path=bilby_input_path,
		start_date=start_date,
		end_date=end_date,
	)
	return run_embedding_match(
		gta_filtered=gta_filtered,
		bilby_filtered=bilby_filtered,
		combined_match_input_path=combined_match_input_path,
		output_dir=output_dir,
		cache_dir=cache_dir,
		device=device,
		batch_size=batch_size,
		window_days=window_days,
		similarity_threshold=similarity_threshold,
		max_candidates_per_query=max_candidates_per_query,
		similarity_chunk_size=similarity_chunk_size,
		show_progress_bar=show_progress_bar,
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run title embedding matching and export all candidates with cosine similarity above threshold."
	)
	parser.add_argument(
		"--gta-input",
		type=Path,
		default=DEFAULT_GTA_INPUT,
		help="Prepared GTA parquet file.",
	)
	parser.add_argument(
		"--bilby-input",
		type=Path,
		default=DEFAULT_BILBY_INPUT,
		help="Combined match bilby unmatched parquet file.",
	)
	parser.add_argument(
		"--combined-match-input",
		type=Path,
		default=DEFAULT_COMBINED_MATCH_INPUT,
		help="Combined exact matched parquet file to merge into embedding matched output.",
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
		help="Directory for embedding match exports.",
	)
	parser.add_argument(
		"--cache-dir",
		type=Path,
		default=DEFAULT_CACHE_DIR,
		help="Directory for cached Bilby embeddings.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default=DEFAULT_DEVICE,
		help="SentenceTransformer device, for example cpu or cuda.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=DEFAULT_BATCH_SIZE,
		help="Batch size used during embedding encoding.",
	)
	parser.add_argument(
		"--window-days",
		type=int,
		default=DEFAULT_WINDOW_DAYS,
		help="Use Bilby candidates within this many days before and after each GTA published date.",
	)
	parser.add_argument(
		"--similarity-threshold",
		type=float,
		default=DEFAULT_SIMILARITY_THRESHOLD,
		help="Export matches with cosine similarity strictly greater than this threshold.",
	)
	parser.add_argument(
		"--max-candidates-per-query",
		type=int,
		default=DEFAULT_MAX_CANDIDATES_PER_QUERY,
		help="Apply a cheap lexical prefilter and cap the Bilby candidates retained per GTA title before embedding. Set 0 to disable.",
	)
	parser.add_argument(
		"--similarity-chunk-size",
		type=int,
		default=DEFAULT_SIMILARITY_CHUNK_SIZE,
		help="Candidate chunk size for similarity computation.",
	)
	parser.add_argument(
		"--hide-progress-bar",
		action="store_true",
		help="Disable sentence-transformers progress bars.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_embedding_match_from_files(
		gta_input_path=args.gta_input,
		bilby_input_path=args.bilby_input,
		combined_match_input_path=args.combined_match_input,
		start_date=pd.Timestamp(args.start_date),
		end_date=pd.Timestamp(args.end_date),
		output_dir=args.output_dir,
		cache_dir=args.cache_dir,
		device=args.device,
		batch_size=args.batch_size,
		window_days=args.window_days,
		similarity_threshold=args.similarity_threshold,
		max_candidates_per_query=args.max_candidates_per_query,
		similarity_chunk_size=args.similarity_chunk_size,
		show_progress_bar=not args.hide_progress_bar,
	)


if __name__ == "__main__":
	main()
