import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple
from urllib.parse import urlparse, urlunparse

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from selenium import webdriver
from selenium.common.exceptions import NoSuchWindowException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Configure default input/output locations and runtime scrape behavior.
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_XLSX = BASE_DIR / "Data/China_GTA_Downloaded/interventions.xlsx"
OUTPUT_XLSX = BASE_DIR / "Data/China_GTA_Source/interventions_sources.xlsx"

SELENIUM_HEADLESS = os.getenv("GTA_SELENIUM_HEADLESS", "0") == "1"
# Default to manual login flow so the script pauses before batch navigation.
SELENIUM_MANUAL_LOGIN = os.getenv("GTA_SELENIUM_MANUAL_LOGIN", "1") == "1"
SELENIUM_DEBUG = os.getenv("GTA_SELENIUM_DEBUG", "0") == "1"
PAGE_LOAD_TIMEOUT = 20
SCRAPE_DELAY_SECONDS = 0.5
CHECKPOINT_INTERVAL = 20


@dataclass(frozen=True)
class SourceRecord:
	# Store normalized source-level fields extracted from one state-act page.
    main_title: str
    source_title: str
    source_url: str
    site_root_url: str
    published_date: str
    summary: str
    state_act_url: str



def resolve_site_root_url(url: str) -> str:
    # Expose site root extraction through a public helper for downstream use.
    return _site_root_url(url)


# ======================== Selenium Setup ========================
def _create_selenium_driver() -> tuple[object, str]:
	# Build a Chrome driver with an isolated temporary user profile.
	options = Options()
	temp_profile_dir = tempfile.mkdtemp(prefix="gta-selenium-")
	if SELENIUM_HEADLESS and not SELENIUM_MANUAL_LOGIN:
		# Run without a visible browser window when headless mode is enabled.
		options.add_argument("--headless=new")
	elif SELENIUM_HEADLESS and SELENIUM_MANUAL_LOGIN:
		# Manual login requires a visible browser window.
		print("GTA_SELENIUM_MANUAL_LOGIN=1 detected, disabling headless mode for login.")
	options.add_argument("--disable-gpu")
	options.add_argument("--no-sandbox")
	options.add_argument("--window-size=1280,900")
	# Persist profile files in a temp directory so sessions can be cleaned safely.
	options.add_argument(f"--user-data-dir={temp_profile_dir}")
	return webdriver.Chrome(options=options), temp_profile_dir


# ======================== URL Helpers ========================
def _normalize_url(url: str) -> str:
	# Normalize protocol-relative URLs and trim surrounding whitespace.
	url = url.strip()
	if url.startswith("//"):
		return f"https:{url}"
	return url


def _site_root_url(url: str) -> str:
	# Convert a full URL into its site root for domain-level grouping.
	parsed_url = urlparse(_normalize_url(url))
	return urlunparse((parsed_url.scheme, parsed_url.netloc, "/", "", "", ""))


# ======================== Date Parsing ========================
def _format_published_date(year: str, month: str, day: str) -> str:
	# Standardize parsed date components into YYYY-MM-DD format.
	return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _extract_date_from_text(text: str) -> str:
	# Exit early when source text is empty.
	if not text:
		return ""

	# Map English month names so natural-language dates can be normalized.
	month_lookup = {
		"january": "01",
		"jan": "01",
		"february": "02",
		"feb": "02",
		"march": "03",
		"mar": "03",
		"april": "04",
		"apr": "04",
		"may": "05",
		"june": "06",
		"jun": "06",
		"july": "07",
		"jul": "07",
		"august": "08",
		"aug": "08",
		"september": "09",
		"sep": "09",
		"sept": "09",
		"october": "10",
		"oct": "10",
		"november": "11",
		"nov": "11",
		"december": "12",
		"dec": "12",
	}
	month_name_pattern = (
		r"January|February|March|April|May|June|July|August|September|October|November|December"
		r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
	)

	# Prefer explicit citation-style dates such as "retrieved on 27 May 2024".
	retrieved_match = re.search(
		rf"\bretrieved\s+on\s+(\d{{1,2}})\s+({month_name_pattern}),?\s+((?:19|20)\d{{2}})\b",
		text,
		flags=re.IGNORECASE,
	)
	if retrieved_match:
		day, month_name, year = retrieved_match.groups()
		return _format_published_date(year, month_lookup[month_name.lower()], day)

	# Support "last accessed on 13 May, 2025" citation style.
	last_accessed_match = re.search(
		rf"\blast\s+accessed\s+on\s+(\d{{1,2}})\s+({month_name_pattern}),?\s+((?:19|20)\d{{2}})\b",
		text,
		flags=re.IGNORECASE,
	)
	if last_accessed_match:
		day, month_name, year = last_accessed_match.groups()
		return _format_published_date(year, month_lookup[month_name.lower()], day)

	# Then handle parenthesized dates such as "(28 May 2024)".
	parenthesized_match = re.search(
		rf"\((\d{{1,2}})\s+({month_name_pattern}),?\s+((?:19|20)\d{{2}})\)",
		text,
		flags=re.IGNORECASE,
	)
	if parenthesized_match:
		day, month_name, year = parenthesized_match.groups()
		return _format_published_date(year, month_lookup[month_name.lower()], day)

	# General English format fallback such as "2 January 2025".
	english_match = re.search(
		rf"\b(\d{{1,2}})\s+({month_name_pattern}),?\s+((?:19|20)\d{{2}})\b",
		text,
		flags=re.IGNORECASE,
	)
	if english_match:
		day, month_name, year = english_match.groups()
		return _format_published_date(year, month_lookup[month_name.lower()], day)

	# Handle slash-separated dates such as "10/01/2024" and "10-01-2024".
	slash_date_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-]((?:19|20)\d{2})\b", text)
	if slash_date_match:
		first, second, year = slash_date_match.groups()
		first_int = int(first)
		second_int = int(second)
		# Default to day/month/year for ambiguous values; swap when only month/day is valid.
		if first_int <= 12 and second_int > 12:
			month = first
			day = second
		else:
			day = first
			month = second
		try:
			return _format_published_date(year, month, day)
		except ValueError:
			pass

	# Then try numeric formats with separators and compact YYYYMMDD values.
	patterns = [
		r"((?:19|20)\d{2})(?:-|/|\.|年)\s?(\d{1,2})(?:-|/|\.|月)\s?(\d{1,2})(?:日)?",
		r"((?:19|20)\d{2})(\d{2})(\d{2})",
	]
	for pattern in patterns:
		# Return the first valid date that can be normalized.
		try:
			match = re.search(pattern, text)
		except re.PatternError as exc:
			# Keep scraping even if one pattern is malformed.
			if SELENIUM_DEBUG:
				print(f"Skipping invalid date regex pattern: {pattern} ({exc})")
			continue
		if not match:
			continue
		year, month, day = match.groups()
		try:
			return _format_published_date(year, month, day)
		except ValueError:
			# Skip impossible dates and continue searching other patterns.
			continue
	# Return empty string when no date signature is found.
	return ""


def extract_published_date(source_text: str) -> str:
	# Provide a public wrapper used by record construction.
	return _extract_date_from_text(source_text)


def build_source_record(
	main_title: str,
	title: str,
	source_url: str,
	summary: str = "",
	state_act_url: str = "",
	announcement_date: str = "",
) -> SourceRecord | None:
	# Normalize and validate source fields before creating a data record.
	clean_main_title = main_title or ""
	clean_title = title or ""
	normalized_source_url = _normalize_url(source_url)

	# Keep the original extracted title text without additional cleaning.
	resolved_title = clean_title

	# Prefer date extracted from the original text, then announcement fallback.
	published_date = extract_published_date(clean_title)
	if not published_date and announcement_date:
		published_date = extract_published_date(announcement_date)

	# Build the immutable record consumed by deduplication and export steps.
	return SourceRecord(
		main_title=clean_main_title,
		source_title=resolved_title,
		source_url=normalized_source_url,
		site_root_url=resolve_site_root_url(normalized_source_url),
		published_date=published_date,
		summary=summary,
		state_act_url=state_act_url,
	)


# ======================== Page Fetching ========================
def _fetch_html_with_selenium(url: str, driver: object | None = None) -> str:
	# Create a temporary driver only when one is not passed by the caller.
	owned_driver = driver is None
	temp_profile_dir: str | None = None
	if owned_driver:
		driver, temp_profile_dir = _create_selenium_driver()

	try:
		assert driver is not None
		# Open the state-act page and optionally emit debug information.
		driver.get(url)
		if SELENIUM_DEBUG:
			print(f"After first get: {driver.current_url} | {driver.title}")
		if "globaltradealert.org" not in driver.current_url:
			# Force navigation back to GTA when intermediate redirects appear.
			driver.execute_script("window.location.href = arguments[0];", url)
			time.sleep(2)
			if SELENIUM_DEBUG:
				print(f"After script nav: {driver.current_url} | {driver.title}")
		# Wait until page routing is stable and Source section is present.
		WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
			lambda d: "globaltradealert.org" in d.current_url
		)
		WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
			EC.presence_of_element_located((By.XPATH, "//*[normalize-space()='Source']"))
		)
		return driver.page_source
	finally:
		if owned_driver:
			# Close and clean temporary profile only for internally owned drivers.
			driver.quit()
			if temp_profile_dir is not None:
				shutil.rmtree(temp_profile_dir, ignore_errors=True)


def _ensure_manual_login_session(driver: object, first_url: str) -> None:
	if not SELENIUM_MANUAL_LOGIN:
		return

	# Open a real GTA page and block until the Source section becomes visible.
	driver.get(first_url)
	while True:
		print(f"Manual login mode active. Browser opened at: {driver.current_url}")
		input("Please sign in to GTA in the browser, then press Enter to continue...")

		# Refresh target page after login and verify access by waiting for Source section.
		driver.get(first_url)
		try:
			WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
				lambda d: "globaltradealert.org" in d.current_url
			)
			WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
				EC.presence_of_element_located((By.XPATH, "//*[normalize-space()='Source']"))
			)
			print("Login confirmed. Starting scrape...", flush=True)
			break
		except TimeoutException:
			print("Login is not confirmed yet. Please complete sign-in and press Enter again.", flush=True)


# ======================== Input Reading ========================
def _read_state_act_urls(input_path: Path) -> List[str]:
	# Validate input file existence before attempting to read it.
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	# Read Excel rows and keep a deduplicated list of state-act URLs.
	df = pd.read_excel(input_path)
	urls: List[str] = []
	seen_urls: Set[str] = set()

	if "State Act URL" in df.columns:
		# Ignore empty values and preserve first-seen URL order.
		series = df["State Act URL"].dropna()
		for value in series:
			url = str(value).strip()
			if not url or url in seen_urls:
				continue
			seen_urls.add(url)
			urls.append(url)
	return urls


# ======================== HTML Section Lookup ========================
def _extract_labeled_section_nodes(soup: BeautifulSoup, label: str) -> List[Tag]:
	# Locate nodes whose text exactly matches a section label such as Source.
	section_nodes: List[Tag] = []
	label_lower = label.lower()
	for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "strong", "label", "span", "p", "div"]):
		if tag.get_text(strip=True).lower() == label_lower:
			section_nodes.append(tag)
	return section_nodes


def _extract_labeled_section_text(soup: BeautifulSoup, label: str) -> str:
	# Try nearby siblings first, then wider sibling traversal as a fallback.
	for node in _extract_labeled_section_nodes(soup, label):
		content = node.find_next_sibling() if isinstance(node, Tag) else None
		if content is None and node.parent is not None:
			content = node.parent.find_next_sibling()
		if isinstance(content, Tag):
			# Return normalized text from direct section content when available.
			text = re.sub(r"\s+", " ", content.get_text(" ", strip=True)).strip()
			if text:
				return text

		parent = node.parent if node.parent is not None else node
		if isinstance(parent, Tag):
			parts: List[str] = []
			for sibling in parent.find_next_siblings():
				# Stop at next major heading to avoid crossing section boundaries.
				if sibling.name in {"h1", "h2", "h3", "h4", "h5"}:
					break
				text = sibling.get_text(" ", strip=True)
				if text:
					parts.append(text)
			if parts:
				return re.sub(r"\s+", " ", " ".join(parts)).strip()
	return ""


def _extract_sources_from_container(container: Tag) -> List[Tuple[str, str]]:
	# Parse one source container into (title, url) pairs in document order.
	"""Extract (title, url) pairs by walking the container's immediate children.

	This captures the text that precedes each link as the title, which matches
	GTA's Source section structure (text + link + <br> separators).
	"""
	sources: List[Tuple[str, str]] = []
	current_text: List[str] = []

	for child in container.children:
		if isinstance(child, NavigableString):
			# Buffer free text so it becomes the title of the next anchor tag.
			text = str(child).strip()
			if text:
				current_text.append(text)
			continue

		if isinstance(child, Tag) and child.name == "a" and child.get("href"):
			# Emit one source pair when a link is encountered.
			url = child.get("href", "").strip()
			title = " ".join(current_text).strip()
			current_text = []
			if url:
				sources.append((title, url))
			continue

		if isinstance(child, Tag) and child.name == "br":
			# Preserve spacing between text fragments split by line breaks.
			if current_text:
				current_text.append(" ")
			continue

		if isinstance(child, Tag):
			# Capture text from non-anchor tags that still belong to source title.
			text = child.get_text(" ", strip=True)
			if text:
				current_text.append(text)

	return sources


def _extract_sources_from_section(section_root: Tag) -> List[Tuple[str, str]]:
	# Traverse nested tags and parse only containers with direct link children.
	"""Find the innermost containers that directly hold source text and links."""
	sources: List[Tuple[str, str]] = []
	for candidate in section_root.find_all(True):
		if candidate.find("a", href=True, recursive=False):
			sources.extend(_extract_sources_from_container(candidate))
	return sources


def parse_sources_from_html(html: str) -> List[Tuple[str, str]]:
	# Build parse tree and find labeled Source sections in the page.
	soup = BeautifulSoup(html, "html.parser")

	source_nodes = _extract_labeled_section_nodes(soup, "Source")

	# Aggregate source links from local section content and fallback siblings.
	links: List[Tuple[str, str]] = []
	for node in source_nodes:
		content = None
		if isinstance(node, Tag):
			content = node.find_next_sibling()
		if content is None and node.parent is not None:
			content = node.parent.find_next_sibling()
		if isinstance(content, Tag):
			links.extend(_extract_sources_from_section(content))
			continue

		# Fallback to scanning subsequent siblings until a new heading is reached.
		parent = node.parent if node.parent is not None else node
		if isinstance(parent, Tag):
			for sibling in parent.find_next_siblings():
				if sibling.name in {"h1", "h2", "h3", "h4", "h5"}:
					break
				links.extend(_extract_sources_from_section(sibling))

	return links


def extract_summary_from_html(html: str) -> str:
	# Extract Announcement summary using a structured description-panel lookup.
	soup = BeautifulSoup(html, "html.parser")
	for node in _extract_labeled_section_nodes(soup, "Announcement"):
		panel = node.find_parent(
			lambda tag: isinstance(tag, Tag)
			and tag.name == "div"
			and tag.find(
				lambda child: isinstance(child, Tag)
				and child.name == "div"
				and any("description" == class_name or class_name.startswith("description-") for class_name in (child.get("class") or []))
			)
		)
		if isinstance(panel, Tag):
			# Prefer the dedicated description block when present.
			description = panel.find(
				lambda tag: isinstance(tag, Tag)
				and tag.name == "div"
				and any("description" == class_name or class_name.startswith("description-") for class_name in (tag.get("class") or []))
			)
			if isinstance(description, Tag):
				text = re.sub(r"\s+", " ", description.get_text(" ", strip=True)).strip()
				if text:
					return text
	# Fall back to generic labeled section text extraction.
	return _extract_labeled_section_text(soup, "Announcement")


def extract_announcement_date_from_html(html: str) -> str:
	# Extract announcement badge date from the announcement header row when present.
	soup = BeautifulSoup(html, "html.parser")
	for row in soup.select("div.flex.items-center.justify-between"):
		label_node = row.find(
			"p",
			string=lambda value: isinstance(value, str) and value.strip().lower() == "announcement",
		)
		if label_node is None:
			continue
		for candidate in row.find_all("p"):
			if candidate is label_node:
				continue
			candidate_text = candidate.get_text(" ", strip=True)
			candidate_date = extract_published_date(candidate_text)
			if candidate_date:
				return candidate_date
	return ""


def extract_main_title_from_html(html: str) -> str:
	# Extract page-level main title from <h1 class="title-page"> for state-act context.
	soup = BeautifulSoup(html, "html.parser")
	main_title_node = soup.find("h1", class_=lambda value: value and "title-page" in value)
	if isinstance(main_title_node, Tag):
		return re.sub(r"\s+", " ", main_title_node.get_text(" ", strip=True)).strip()
	return ""


def scrape_sources_from_state_act(url: str, driver: object) -> tuple[str, str, str, List[Tuple[str, str]]]:
	# Fetch page HTML, then extract page-level title, announcement date, summary, and source link pairs.
	html = _fetch_html_with_selenium(url, driver=driver)
	return (
		extract_main_title_from_html(html),
		extract_announcement_date_from_html(html),
		extract_summary_from_html(html),
		parse_sources_from_html(html),
	)


# ======================== Deduplication and Export ========================
def dedupe_sources_by_title_and_url(sources: Iterable[SourceRecord]) -> List[SourceRecord]:
	# Deduplicate records by title-url composite key while preserving order.
	seen: Set[Tuple[str, str]] = set()
	unique: List[SourceRecord] = []
	for source in sources:
		key = (source.source_title, source.source_url)
		if key in seen:
			continue
		seen.add(key)
		unique.append(source)
	return unique


def _is_missing_published_date(value: str | None) -> bool:
	# Treat empty values and explicit null markers as missing dates.
	if value is None:
		return True
	text = str(value).strip()
	return text == "" or text.lower() == "null"


def clean_published_date_by_state_act(sources: List[SourceRecord]) -> List[SourceRecord]:
	# Fill missing dates from same state-act first, then from title text as fallback.
	state_act_date_map: dict[str, str] = {}
	for source in sources:
		if _is_missing_published_date(source.published_date):
			continue
		state_act_url = (source.state_act_url or "").strip()
		if not state_act_url or state_act_url.lower() == "null":
			continue
		state_act_date_map.setdefault(state_act_url, source.published_date.strip())

	cleaned_sources: List[SourceRecord] = []
	for source in sources:
		if not _is_missing_published_date(source.published_date):
			cleaned_sources.append(source)
			continue

		resolved_date = ""
		state_act_url = (source.state_act_url or "").strip()
		if state_act_url and state_act_url.lower() != "null":
			resolved_date = state_act_date_map.get(state_act_url, "")

		if not resolved_date:
			resolved_date = extract_published_date(source.source_title)
		if not resolved_date:
			resolved_date = extract_published_date(source.main_title)

		if not resolved_date:
			cleaned_sources.append(source)
			continue

		cleaned_sources.append(
			SourceRecord(
				main_title=source.main_title,
				source_title=source.source_title,
				source_url=source.source_url,
				site_root_url=source.site_root_url,
				published_date=resolved_date,
				summary=source.summary,
				state_act_url=source.state_act_url,
			)
		)

	return cleaned_sources


def export_sources_to_excel(sources: List[SourceRecord], output_path: Path) -> None:
	# Transform records into a tabular structure and write to Excel.
    df = pd.DataFrame(
        [
            {
				"main_title": source.main_title,
                "source_title": source.source_title,
                "source_url": source.source_url,
                "site_root_url": source.site_root_url,
                "published_date": source.published_date or "null",
                "summary": source.summary or "null",
                "state_act_url": source.state_act_url or "null",
            }
            for source in sources
        ]
    )
	# Persist a single worksheet containing normalized source metadata.
    df.to_excel(output_path, index=False)


def export_sources_to_parquet(sources: List[SourceRecord], output_path: Path) -> None:
	# Persist the same normalized source metadata in parquet format.
	df = pd.DataFrame(
		[
			{
				"main_title": source.main_title,
				"source_title": source.source_title,
				"source_url": source.source_url,
				"site_root_url": source.site_root_url,
				"published_date": source.published_date or "null",
				"summary": source.summary or "null",
				"state_act_url": source.state_act_url or "null",
			}
			for source in sources
		]
	)
	df.to_parquet(output_path, index=False)


def _resolve_non_overwrite_output_path(base_output_xlsx: Path) -> Path:
	# Choose a new filename with (n) suffix when xlsx/parquet output already exists.
	base_output_xlsx.parent.mkdir(parents=True, exist_ok=True)
	parquet_base = base_output_xlsx.with_suffix(".parquet")

	if not base_output_xlsx.exists() and not parquet_base.exists():
		return base_output_xlsx

	stem = base_output_xlsx.stem
	suffix = base_output_xlsx.suffix
	for index in range(1, 10000):
		candidate_xlsx = base_output_xlsx.with_name(f"{stem}({index}){suffix}")
		candidate_parquet = candidate_xlsx.with_suffix(".parquet")
		if not candidate_xlsx.exists() and not candidate_parquet.exists():
			return candidate_xlsx

	raise RuntimeError("Unable to find a non-overwriting output filename.")


def export_current_results(sources: List[SourceRecord], output_path: Path) -> int:
	# Clean missing dates, then save deduplicated snapshots in both xlsx and parquet formats.
	cleaned_sources = clean_published_date_by_state_act(sources)
	unique_sources = dedupe_sources_by_title_and_url(cleaned_sources)
	export_sources_to_excel(unique_sources, output_path)
	export_sources_to_parquet(unique_sources, output_path.with_suffix(".parquet"))
	return len(unique_sources)


# ======================== Main Orchestration ========================
def main() -> None:
	# Print effective runtime paths to make config sync explicit on every run.
	print(f"Using INPUT_XLSX: {INPUT_XLSX}", flush=True)
	print(f"Using OUTPUT_XLSX: {OUTPUT_XLSX}", flush=True)

	# Phase 1: Load all state-act URLs from the configured input file.
	state_act_urls = _read_state_act_urls(INPUT_XLSX)
	if not state_act_urls:
		raise RuntimeError("No State Act URLs found in the input Excel file.")

	# Initialize in-memory collectors and progress counters.
	all_sources: List[SourceRecord] = []
	last_completed_index = 0
	total_urls = len(state_act_urls)
	resolved_output_xlsx = _resolve_non_overwrite_output_path(OUTPUT_XLSX)
	if resolved_output_xlsx != OUTPUT_XLSX:
		print(
			f"Output already exists. Using {resolved_output_xlsx} and {resolved_output_xlsx.with_suffix('.parquet')} instead.",
			flush=True,
		)
	print(f"Starting scrape for {total_urls} state-act pages", flush=True)

	# Reuse one browser session across pages to reduce startup overhead.
	driver, temp_profile_dir = _create_selenium_driver()
	try:
		# In manual login mode, block here until the user completes one-time sign-in.
		_ensure_manual_login_session(driver, state_act_urls[0])

		# Phase 2: Scrape each state-act page and accumulate valid source records.
		for idx, state_act_url in enumerate(state_act_urls, start=1):
			print(f"[{idx}/{total_urls}] Scraping {state_act_url}", flush=True)

			# Fetch page content and parse summary plus raw source tuples.
			try:
				main_title, announcement_date, summary, sources = scrape_sources_from_state_act(state_act_url, driver=driver)
			except NoSuchWindowException:
				# Recover when the browser window is closed manually during a long run.
				print("Browser window was closed. Reopening browser and retrying current page...", flush=True)
				try:
					driver.quit()
				except Exception:
					pass
				shutil.rmtree(temp_profile_dir, ignore_errors=True)
				driver, temp_profile_dir = _create_selenium_driver()
				_ensure_manual_login_session(driver, state_act_url)
				main_title, announcement_date, summary, sources = scrape_sources_from_state_act(state_act_url, driver=driver)

			records = []
			for title, source_url in sources:
				# Validate and normalize each extracted source tuple.
				record = build_source_record(
					main_title,
					title,
					source_url,
					summary=summary,
					state_act_url=state_act_url,
					announcement_date=announcement_date,
				)
				records.append(record)

			# Keep valid records and report incremental progress.
			all_sources.extend(records)
			last_completed_index = idx
			print(
				f"[{idx}/{total_urls}] Kept {len(records)} sources; total collected: {len(all_sources)}",
				flush=True,
			)

			if idx % CHECKPOINT_INTERVAL == 0:
				# Periodically export deduplicated snapshots during long runs.
				saved_count = export_current_results(all_sources, resolved_output_xlsx)
				print(f"Checkpoint saved after {idx} pages: {saved_count} unique sources", flush=True)
				time.sleep(SCRAPE_DELAY_SECONDS)

	except KeyboardInterrupt:
		# Phase 3: Save partial results before honoring user interruption.
		saved_count = export_current_results(all_sources, resolved_output_xlsx)
		print(
			f"Interrupted after {last_completed_index} pages. "
			f"Saved {saved_count} unique sources to {resolved_output_xlsx} and {resolved_output_xlsx.with_suffix('.parquet')}.",
			flush=True,
		)
		raise
	except Exception:
		# Save partial results before propagating unexpected scraper errors.
		if all_sources:
			saved_count = export_current_results(all_sources, resolved_output_xlsx)
			print(
				f"Stopped after {last_completed_index} pages due to an error. "
				f"Saved {saved_count} unique sources to {resolved_output_xlsx} and {resolved_output_xlsx.with_suffix('.parquet')}.",
				flush=True,
			)
		raise
	finally:
		# Always release browser resources and remove temporary profile files.
		driver.quit()
		shutil.rmtree(temp_profile_dir, ignore_errors=True)

	# Phase 4: Final deduplicated export and completion summary.
	saved_count = export_current_results(all_sources, resolved_output_xlsx)
	print(f"Total sources collected before deduplication: {len(all_sources)}", flush=True)
	print(
		f"Exported {saved_count} unique sources to {resolved_output_xlsx} and {resolved_output_xlsx.with_suffix('.parquet')}",
		flush=True,
	)

if __name__ == "__main__":
	main()



