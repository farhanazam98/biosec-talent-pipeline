import argparse
import asyncio
import csv
import glob
import json
import os
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone

import trafilatura
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

WORK_QUEUE = "data/work_queue.csv"
RAW_DIR = "data/raw"
CONCURRENCY = 10  # max simultaneous fetches

# Phrases that indicate the page blocked the request rather than serving real content.
BLOCK_SIGNALS = [
    "suspicious activity",
    "access denied",
    "403 forbidden",
    "blocked",
    "captcha",
    "verify you are human",
    "are you a robot",
    "enable javascript",
    "please enable cookies",
    "ddos protection",
    "checking your browser",
    "just a moment",  # Cloudflare
]


def url_to_filename(url: str) -> str:
    slug = re.sub(r"https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:120] + ".json"


def is_blocked(text: str) -> bool:
    """Return True if the extracted text looks like a bot-block page."""
    lower = text.lower()
    return any(signal in lower for signal in BLOCK_SIGNALS)


def _http_get_sync(url: str) -> tuple:
    """Blocking HTTP fetch — call via run_in_executor to avoid stalling the event loop."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; biosec-pipeline/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace"), resp.status
    except urllib.error.HTTPError as e:
        return None, e.code
    except Exception:
        return None, 0


async def fetch_with_trafilatura(url: str) -> tuple:
    """Returns (raw_text, fetch_method, fetch_status)."""
    loop = asyncio.get_event_loop()
    html, status_code = await loop.run_in_executor(None, _http_get_sync, url)

    if status_code == 0 or status_code >= 400:
        return "", "trafilatura", "failed"
    if not html:
        return "", "trafilatura", "failed"

    text = await loop.run_in_executor(None, lambda: trafilatura.extract(html) or "")
    if not text.strip():
        return "", "trafilatura", "partial"
    if is_blocked(text):
        return text, "trafilatura", "partial"
    return text, "trafilatura", "ok"


async def fetch_with_playwright(url: str) -> tuple:
    """Returns (raw_text, fetch_method, fetch_status)."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            response = await page.goto(url, timeout=30000)
            status_code = response.status if response else 0
            html = await page.content()
            await browser.close()
    except Exception:
        return "", "playwright", "failed"

    if status_code >= 400:
        return "", "playwright", "failed"

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, lambda: trafilatura.extract(html) or "")
    if not text.strip():
        return "", "playwright", "partial"
    if is_blocked(text):
        return text, "playwright", "partial"
    return text, "playwright", "ok"


async def fetch(url: str) -> tuple:
    """Try trafilatura first, fall back to playwright."""
    raw_text, method, status = await fetch_with_trafilatura(url)
    if status == "ok":
        return raw_text, method, status
    return await fetch_with_playwright(url)


# Normalize old work queue category names to current taxonomy
HINT_TYPE_MAPPING = {
    "formal_training": "formal_training",
    "fellowship_competition": "non_degree_structured",
    "gov_multilateral": "gov_institutional",
}


def build_record(row: dict, raw_text: str, fetch_method: str, fetch_status: str) -> dict:
    raw_type = row["type_hint"]
    return {
        "url": row["url"],
        "hints": {
            "name": row["name_hint"],
            "lead_org": row["lead_org_hint"],
            "country": row["country_hint"],
            "type": HINT_TYPE_MAPPING.get(raw_type, raw_type),
            "active_status": row["active_status_hint"],
            "region": row["region_hint"],
        },
        "source_doc_id": row["source_doc_id"],
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetch_method": fetch_method,
        "fetch_status": fetch_status,
        "raw_text": raw_text,
    }


async def process_row(sem: asyncio.Semaphore, row: dict, index: int, total: int) -> None:
    async with sem:
        url = row["url"]
        raw_text, fetch_method, fetch_status = await fetch(url)
        record = build_record(row, raw_text, fetch_method, fetch_status)
        out_path = os.path.join(RAW_DIR, url_to_filename(url))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"[{fetch_status}] ({index}/{total}) {url}")


def cached_status(path: str) -> str:
    """Return the fetch_status of a cached JSON, or empty string if unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("fetch_status", "") or ""
    except Exception:
        return ""


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all URLs, ignoring cached fetch results",
    )
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    with open(WORK_QUEUE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Build filename -> row map (dedupes URLs that appear twice in the work queue —
    # Stage 1 writes one JSON per URL anyway).
    expected_files = {}
    for row in rows:
        expected_files.setdefault(url_to_filename(row["url"]), row)

    existing_files = {
        os.path.basename(p) for p in glob.glob(os.path.join(RAW_DIR, "*.json"))
    }

    # Delete orphan JSONs whose URL is no longer in the work queue, so Stage 2
    # doesn't reprocess dead records.
    orphans = existing_files - set(expected_files)
    for fname in orphans:
        os.remove(os.path.join(RAW_DIR, fname))
    if orphans:
        print(f"Deleted {len(orphans)} orphan files (URL no longer in work queue)")

    # Skip URLs whose cached JSON already has fetch_status: ok (unless --force).
    # Re-fetch missing/failed/partial so previously-failed URLs get another chance.
    rows_to_fetch = []
    cached_ok = 0
    for fname, row in expected_files.items():
        path = os.path.join(RAW_DIR, fname)
        if not args.force and os.path.exists(path) and cached_status(path) == "ok":
            cached_ok += 1
            continue
        rows_to_fetch.append(row)

    if args.force and existing_files:
        print(f"--force: ignoring {len(existing_files - orphans)} cached entries")
    elif cached_ok:
        print(f"Skipping {cached_ok} URLs with cached fetch_status: ok")

    if not rows_to_fetch:
        print(f"Nothing to fetch. {cached_ok} cached entries in {RAW_DIR}/")
        return

    print(f"Fetching {len(rows_to_fetch)} URLs (concurrency={CONCURRENCY})")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        process_row(sem, row, i, len(rows_to_fetch))
        for i, row in enumerate(rows_to_fetch, 1)
    ]
    await asyncio.gather(*tasks)

    total = cached_ok + len(rows_to_fetch)
    print(f"\nDone. {cached_ok} cached + {len(rows_to_fetch)} fetched = {total} files in {RAW_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
