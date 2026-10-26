"""
fetch_cma.py — Cleveland Museum of Art Open Access API data collector.

Fetches artwork records from the CMA Open Access API, focusing on CC0-licensed
works with images and substantive descriptions. Handles pagination, rate limiting,
and incomplete metadata gracefully.

API Docs: https://openaccess-api.clevelandart.org/
License: CC0 (Creative Commons Zero) — unrestricted use
No API key required.
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "https://openaccess-api.clevelandart.org/api/artworks/"
DEFAULT_LIMIT = 100          # records per API page (max 1000)
DEFAULT_DELAY = 0.5          # seconds between requests (polite crawling)
MAX_RETRIES = 3              # retries per failed request
RETRY_BACKOFF = 2.0          # exponential backoff multiplier

# Fields we want to preserve from each record
FIELDS_TO_KEEP = [
    "id",
    "accession_number",
    "share_license_status",
    "title",
    "creation_date",
    "creation_date_earliest",
    "creation_date_latest",
    "culture",
    "technique",
    "department",
    "collection",
    "type",
    "measurements",
    "creditline",
    "description",
    "did_you_know",
    "tombstone",
    "provenance",
    "find_spot",
    "url",
    "creators",
    "images",
    "current_location",
    "citations",
    "catalogue_raisonne",
    "exhibitions",
    "inscriptions",
    "copyright",
    "fun_fact",
]

# Directories
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
# API helpers
# ---------------------------------------------------------------------------

def fetch_page(
    skip: int = 0,
    limit: int = DEFAULT_LIMIT,
    cc0: bool = True,
    has_image: bool = True,
    extra_params: Optional[dict] = None,
) -> dict:
    """Fetch a single page of artwork records from the CMA API.

    Args:
        skip: Offset for pagination.
        limit: Number of records to return (max 1000).
        cc0: If True, only return CC0-licensed works.
        has_image: If True, only return works with images.
        extra_params: Additional query parameters.

    Returns:
        Parsed JSON response dict with 'info' and 'data' keys.

    Raises:
        requests.HTTPError: On non-retryable HTTP errors.
    """
    params = {"skip": skip, "limit": limit}
    if cc0:
        params["cc0"] = 1
    if has_image:
        params["has_image"] = 1
    if extra_params:
        params.update(extra_params)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = RETRY_BACKOFF ** attempt
                log.warning(
                    "HTTP %s on attempt %d — retrying in %.1fs",
                    resp.status_code, attempt, wait,
                )
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.RequestException as exc:
            wait = RETRY_BACKOFF ** attempt
            log.warning(
                "Request error on attempt %d: %s — retrying in %.1fs",
                attempt, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch page (skip={skip}) after {MAX_RETRIES} attempts")


def extract_record(raw: dict) -> dict:
    """Extract and normalise a single artwork record.

    Flattens nested creator info and image URLs into simple fields.
    Preserves all original fields listed in FIELDS_TO_KEEP.
    """
    rec = {}

    # Copy simple fields
    for field in FIELDS_TO_KEEP:
        rec[field] = raw.get(field)

    # ---------- Flatten creators ----------
    creators = raw.get("creators") or []
    if creators:
        primary = creators[0]
        rec["creator_name"] = primary.get("description", "")
        rec["creator_role"] = primary.get("role", "")
        rec["creator_bio"] = primary.get("biography", "")
        rec["creator_birth_year"] = primary.get("birth_year")
        rec["creator_death_year"] = primary.get("death_year")
    else:
        rec["creator_name"] = None
        rec["creator_role"] = None
        rec["creator_bio"] = None
        rec["creator_birth_year"] = None
        rec["creator_death_year"] = None

    # ---------- Flatten images ----------
    images = raw.get("images") or {}
    web = images.get("web") or {}
    rec["image_url"] = web.get("url")
    rec["image_width"] = web.get("width")
    rec["image_height"] = web.get("height")

    # Also keep print-quality URL for potential future use
    print_img = images.get("print") or {}
    rec["image_print_url"] = print_img.get("url")

    # ---------- Flatten culture ----------
    culture = raw.get("culture") or []
    rec["culture_flat"] = "; ".join(culture) if culture else None

    # ---------- Museum URL ----------
    rec["museum_url"] = raw.get("url")
    if not rec["museum_url"] and raw.get("accession_number"):
        rec["museum_url"] = f"https://clevelandart.org/art/{raw['accession_number']}"

    return rec


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

def fetch_artworks(
    total_target: int = 50,
    limit: int = DEFAULT_LIMIT,
    delay: float = DEFAULT_DELAY,
    cc0: bool = True,
    has_image: bool = True,
    require_description: bool = True,
) -> list[dict]:
    """Fetch multiple pages of artwork records.

    Args:
        total_target: Target number of records to collect.
        limit: Page size.
        delay: Delay between pages.
        cc0: CC0-only filter.
        has_image: Image-required filter.
        require_description: If True, skip records with no description.

    Returns:
        List of extracted record dicts.
    """
    all_records = []
    skip = 0

    # First request to get total available
    first_page = fetch_page(skip=0, limit=1, cc0=cc0, has_image=has_image)
    total_available = first_page["info"]["total"]
    log.info("Total CC0 artworks with images available: %d", total_available)
    log.info("Target collection size: %d", total_target)

    while len(all_records) < total_target and skip < total_available:
        page_limit = min(limit, total_target - len(all_records) + 200)  # overshoot to handle filtering
        page_data = fetch_page(skip=skip, limit=page_limit, cc0=cc0, has_image=has_image)
        records = page_data.get("data", [])

        if not records:
            log.info("No more records returned at skip=%d", skip)
            break

        for raw in records:
            rec = extract_record(raw)

            # Filter: skip records without description if required
            if require_description and not rec.get("description"):
                continue

            all_records.append(rec)

            if len(all_records) >= total_target:
                break

        skip += len(records)
        log.info(
            "Fetched page: skip=%d, got %d raw, kept %d total",
            skip, len(records), len(all_records),
        )
        time.sleep(delay)

    log.info("Collection complete: %d records gathered", len(all_records))
    return all_records


def save_raw(records: list[dict], filename: str = "cma_artworks_raw.json") -> Path:
    """Save raw records to JSON."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info("Saved %d raw records to %s", len(records), path)
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Fetch a sample of CMA artworks and save to data/raw/."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch CMA artworks")
    parser.add_argument(
        "--target", type=int, default=50,
        help="Number of records to collect (default: 50)"
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Records per API page (default: 100)"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help="Delay between API requests in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--no-description-filter", action="store_true",
        help="Include records without descriptions"
    )
    parser.add_argument(
        "--output", type=str, default="cma_artworks_raw.json",
        help="Output filename (saved to data/raw/)"
    )
    args = parser.parse_args()

    records = fetch_artworks(
        total_target=args.target,
        limit=args.limit,
        delay=args.delay,
        require_description=not args.no_description_filter,
    )
    save_raw(records, args.output)
    print(f"\n[OK] Collected {len(records)} records -> data/raw/{args.output}")


if __name__ == "__main__":
    main()
