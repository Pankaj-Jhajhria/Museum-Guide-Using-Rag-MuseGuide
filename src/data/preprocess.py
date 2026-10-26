"""
preprocess.py — Clean and prepare CMA artwork data for RAG.

Takes raw JSON records from fetch_cma.py and produces a cleaned CSV/JSON
ready for embedding generation. Handles missing data, deduplication,
HTML tag stripping, and image URL validation.
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def strip_html(text: Optional[str]) -> Optional[str]:
    """Remove HTML tags from text."""
    if not text:
        return text
    clean = re.sub(r"<[^>]+>", "", text)
    # Normalise whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean if clean else None


def clean_text_field(text: Optional[str]) -> Optional[str]:
    """Clean a text field: strip HTML, normalise whitespace, handle empties."""
    if text is None:
        return None
    text = strip_html(str(text))
    if not text or text.lower() in ("none", "null", "n/a", ""):
        return None
    return text


# ---------------------------------------------------------------------------
# Record cleaning
# ---------------------------------------------------------------------------

def clean_record(rec: dict) -> dict:
    """Clean a single artwork record."""
    cleaned = {}

    # Core identifiers
    cleaned["id"] = rec.get("id")
    cleaned["accession_number"] = rec.get("accession_number")
    cleaned["share_license_status"] = rec.get("share_license_status", "Unknown")

    # Text fields — clean HTML
    cleaned["title"] = clean_text_field(rec.get("title"))
    cleaned["description"] = clean_text_field(rec.get("description"))
    cleaned["did_you_know"] = clean_text_field(rec.get("did_you_know"))
    cleaned["tombstone"] = clean_text_field(rec.get("tombstone"))

    # Creator
    cleaned["creator_name"] = clean_text_field(rec.get("creator_name"))
    cleaned["creator_role"] = rec.get("creator_role")
    cleaned["creator_bio"] = clean_text_field(rec.get("creator_bio"))
    cleaned["creator_birth_year"] = rec.get("creator_birth_year")
    cleaned["creator_death_year"] = rec.get("creator_death_year")

    # Dates
    cleaned["creation_date"] = clean_text_field(rec.get("creation_date"))
    cleaned["creation_date_earliest"] = rec.get("creation_date_earliest")
    cleaned["creation_date_latest"] = rec.get("creation_date_latest")

    # Classification
    cleaned["culture"] = clean_text_field(rec.get("culture_flat"))
    cleaned["technique"] = clean_text_field(rec.get("technique"))
    cleaned["department"] = rec.get("department")
    cleaned["collection"] = rec.get("collection")
    cleaned["type"] = rec.get("type")

    # Location / measurements
    cleaned["current_location"] = rec.get("current_location")
    cleaned["measurements"] = rec.get("measurements")
    cleaned["creditline"] = rec.get("creditline")

    # URLs
    cleaned["museum_url"] = rec.get("museum_url") or rec.get("url")
    cleaned["image_url"] = rec.get("image_url")
    cleaned["image_print_url"] = rec.get("image_print_url")

    return cleaned


# ---------------------------------------------------------------------------
# Build retrieval text
# ---------------------------------------------------------------------------

def build_retrieval_text(rec: dict) -> str:
    """Compose a single text passage for embedding from cleaned fields.

    This combines title, creator, date, technique, description, and
    did_you_know into one coherent passage that captures the most
    useful information for semantic retrieval.
    """
    parts = []

    if rec.get("title"):
        creator_str = f" by {rec['creator_name']}" if rec.get("creator_name") else ""
        date_str = f" ({rec['creation_date']})" if rec.get("creation_date") else ""
        parts.append(f"{rec['title']}{creator_str}{date_str}.")

    if rec.get("technique"):
        parts.append(f"Medium: {rec['technique']}.")

    if rec.get("culture"):
        parts.append(f"Culture: {rec['culture']}.")

    if rec.get("department"):
        parts.append(f"Department: {rec['department']}.")

    if rec.get("description"):
        parts.append(rec["description"])

    if rec.get("did_you_know"):
        parts.append(f"Did you know? {rec['did_you_know']}")

    if rec.get("creator_bio"):
        parts.append(f"About the artist: {rec['creator_bio']}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_image_url(url: str, timeout: float = 5.0) -> bool:
    """Quick HEAD request to check if an image URL is accessible."""
    if not url:
        return False
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    input_file: str = "cma_artworks_raw.json",
    output_prefix: str = "cma_artworks",
    validate_images: bool = False,
    sample_validate: int = 5,
) -> pd.DataFrame:
    """Full preprocessing pipeline.

    Args:
        input_file: Raw JSON filename in data/raw/.
        output_prefix: Prefix for output files in data/processed/.
        validate_images: Whether to HEAD-check all image URLs.
        sample_validate: Number of random images to validate (if not all).

    Returns:
        Cleaned DataFrame.
    """
    # Load raw data
    raw_path = RAW_DIR / input_file
    log.info("Loading raw data from %s", raw_path)
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)
    log.info("Loaded %d raw records", len(raw_records))

    # Clean records
    cleaned = [clean_record(r) for r in raw_records]
    df = pd.DataFrame(cleaned)

    # Deduplicate by ID
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    dupes = before - len(df)
    if dupes > 0:
        log.warning("Removed %d duplicate records", dupes)

    # Drop records with no title AND no description
    df = df.dropna(subset=["title", "description"], how="all")
    log.info("After cleanup: %d records", len(df))

    # Build retrieval text
    df["retrieval_text"] = df.apply(
        lambda row: build_retrieval_text(row.to_dict()), axis=1
    )

    # Check retrieval text lengths
    df["text_length"] = df["retrieval_text"].str.len()

    # Image validation (sample only by default to save time)
    if validate_images:
        log.info("Validating ALL image URLs (this may take a while)...")
        df["image_valid"] = df["image_url"].apply(validate_image_url)
    elif sample_validate > 0:
        sample_idx = df["image_url"].dropna().sample(
            min(sample_validate, len(df)), random_state=42
        ).index
        df["image_valid"] = None
        for idx in sample_idx:
            url = df.at[idx, "image_url"]
            valid = validate_image_url(url)
            df.at[idx, "image_valid"] = valid
            log.info("Image check: %s -> %s", url[:80], "OK" if valid else "FAIL")

    # Save processed data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = PROCESSED_DIR / f"{output_prefix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    log.info("Saved CSV to %s", csv_path)

    json_path = PROCESSED_DIR / f"{output_prefix}.json"
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    log.info("Saved JSON to %s", json_path)

    return df


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def quality_report(df: pd.DataFrame) -> str:
    """Generate a concise data quality report."""
    lines = [
        "=" * 60,
        "  MuseGuide — Data Quality Report",
        "=" * 60,
        f"  Total records: {len(df)}",
        "",
        "  --- Field Completeness ---",
    ]

    key_fields = [
        "id", "title", "creator_name", "creation_date", "technique",
        "description", "did_you_know", "culture", "department",
        "image_url", "museum_url", "creator_bio",
    ]

    for field in key_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            pct = non_null / len(df) * 100
            lines.append(f"  {field:25s}  {non_null:4d}/{len(df)}  ({pct:5.1f}%)")

    # Licensing
    if "share_license_status" in df.columns:
        lines.append("")
        lines.append("  --- Licensing ---")
        for status, count in df["share_license_status"].value_counts().items():
            lines.append(f"  {status:25s}  {count:4d}")

    # Types
    if "type" in df.columns:
        lines.append("")
        lines.append("  --- Artwork Types (top 10) ---")
        for atype, count in df["type"].value_counts().head(10).items():
            lines.append(f"  {atype:25s}  {count:4d}")

    # Departments
    if "department" in df.columns:
        lines.append("")
        lines.append("  --- Departments (top 10) ---")
        for dept, count in df["department"].value_counts().head(10).items():
            lines.append(f"  {dept:35s}  {count:4d}")

    # Retrieval text statistics
    if "text_length" in df.columns:
        lines.append("")
        lines.append("  --- Retrieval Text Statistics ---")
        lines.append(f"  Mean length:   {df['text_length'].mean():.0f} chars")
        lines.append(f"  Median length: {df['text_length'].median():.0f} chars")
        lines.append(f"  Min length:    {df['text_length'].min():.0f} chars")
        lines.append(f"  Max length:    {df['text_length'].max():.0f} chars")

    # Image validation
    if "image_valid" in df.columns:
        checked = df["image_valid"].notna().sum()
        if checked > 0:
            valid = df["image_valid"].sum()
            lines.append("")
            lines.append("  --- Image URL Validation ---")
            lines.append(f"  Checked: {checked}, Valid: {int(valid)}, Invalid: {checked - int(valid)}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess CMA artwork data")
    parser.add_argument(
        "--input", type=str, default="cma_artworks_raw.json",
        help="Input filename in data/raw/"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="cma_artworks",
        help="Output prefix in data/processed/"
    )
    parser.add_argument(
        "--validate-all-images", action="store_true",
        help="Validate all image URLs (slow)"
    )
    args = parser.parse_args()

    df = preprocess(
        input_file=args.input,
        output_prefix=args.output_prefix,
        validate_images=args.validate_all_images,
    )

    report = quality_report(df)
    print(report)

    # Save report
    report_path = PROCESSED_DIR / "quality_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
