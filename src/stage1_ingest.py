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


def build_record(row: dict, raw_text: str, fetch_method: str, fetch_status: str) -> dict:
    return {
        "url": row["url"],
        "hints": {
            "name": row["name_hint"],
            "lead_org": row["lead_org_hint"],
            "country": row["country_hint"],
            "type": row["type_hint"],
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


async def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    existing = glob.glob(os.path.join(RAW_DIR, "*.json"))
    for path in existing:
        os.remove(path)
    if existing:
        print(f"Cleared {len(existing)} existing files from {RAW_DIR}/")

    with open(WORK_QUEUE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} records from {WORK_QUEUE} (concurrency={CONCURRENCY})")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_row(sem, row, i, len(rows)) for i, row in enumerate(rows, 1)]
    await asyncio.gather(*tasks)

    print(f"\nDone. {len(rows)} files written to {RAW_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
