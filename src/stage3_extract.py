import asyncio
import csv
import glob
import json
import os
import re
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()

CLASSIFIED_DIR = "data/classified"
OUTPUT_CSV = "output/stage3_results.csv"
MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
CONCURRENCY = 5  # max simultaneous Claude API calls — tune up if rate limits allow

FIELDS = [
    "name_and_title",
    "organisation_providing_course",
    "pipeline_type",
    "country",
    "organisation_funding_course",
    "expected_outcomes",
    "syllabus_course_materials",
    "target_audience",
    "financial_support_available",
    "visa_travel_constraints",
    "languages",
    "year_established",
    "income_classification",
    "format",
    "focus_area",
    "ai_risks_content_included",
    "dual_use_risks_content_included",
]

FIELD_DESCRIPTIONS = {
    "name_and_title": "Full official program name",
    "organisation_providing_course": "Host / delivering organization",
    "pipeline_type": "One of: formal_training | fellowship_competition | gov_multilateral",
    "country": (
        "Pipe-delimited list of full country names where the program is delivered (e.g. USA|Canada). "
        "Use 'Global' if not specific to any country or if 40+ countries are covered. "
        "Use 'Regional – <region>' (e.g. 'Regional – Africa') if the program targets a broad region without listing specific countries."
    ),
    "organisation_funding_course": "Funder(s) of the program",
    "expected_outcomes": "Stated learning or career outcomes",
    "syllabus_course_materials": "Topics, modules, or curriculum links",
    "target_audience": "Career stage, background, or nationality requirements for applicants",
    "financial_support_available": (
        "Classify as exactly one of: full | partial | free | none | unknown. "
        "full = tuition, stipend, and/or living costs covered. "
        "partial = some costs covered (travel grant, fee waiver, accommodation only, etc.). "
        "free = no tuition fee but no additional financial support. "
        "none = explicitly no financial support available. "
        "unknown = not mentioned on the page."
    ),
    "visa_travel_constraints": (
        "Classify as exactly one of: yes | no | unknown. "
        "yes = any visa requirement, travel obligation, or nationality restriction is mentioned. "
        "no = page explicitly states no constraints. "
        "unknown = not mentioned on the page."
    ),
    "languages": (
        "Pipe-delimited list of delivery language names in English (e.g. English|French|Spanish). "
        "Omit parenthetical notes such as '(simultaneous interpretation)'."
    ),
    "year_established": "Year the program was founded or first offered",
    "income_classification": "One of: HIC | LMIC | Both",
    "format": "e.g. in-person, online, hybrid, part-time, full-time",
    "focus_area": (
        "Pipe-delimited list of tags from this controlled vocabulary: "
        "biosafety (lab biosafety, biorisk management, containment) | "
        "biosecurity_policy (biosecurity governance, arms control, BWC, regulatory frameworks) | "
        "biodefense (biodefense, CBRN defense, threat response) | "
        "biosurveillance (disease surveillance, early warning, outbreak detection) | "
        "dual_use (dual-use research of concern, DURC) | "
        "laboratory_skills (lab techniques, diagnostics, capacity building) | "
        "pandemic_preparedness (pandemic response, epidemic preparedness) | "
        "health_security (health security systems, infectious disease control) | "
        "one_health (One Health, human-animal-environment interface) | "
        "threat_assessment (risk assessment, strategic analysis, intelligence) | "
        "ai_biosecurity (AI risks in biology, AI biosecurity governance). "
        "Choose only tags that clearly apply. Use a short free-text value only if nothing fits."
    ),
    "ai_risks_content_included": "Y or N — does the program cover AI risks?",
    "dual_use_risks_content_included": "Y or N — does the program cover dual-use risks?",
}

# Hints that map directly to extracted fields for conflict detection
HINT_TO_FIELD = {
    "name": "name_and_title",
    "lead_org": "organisation_providing_course",
    "country": "country",
    "type": "pipeline_type",
}

TOOL_DEFINITION = {
    "name": "extract_program_fields",
    "description": "Extract structured fields about a biosecurity training program from its webpage content.",
    "input_schema": {
        "type": "object",
        "properties": {
            field: {
                "type": "object",
                "description": FIELD_DESCRIPTIONS[field],
                "properties": {
                    "value": {"type": "string", "description": "Extracted value, or empty string if not found"},
                    "evidence": {"type": "string", "description": "Verbatim snippet from the page supporting this value, or empty string if not found"},
                },
                "required": ["value", "evidence"],
            }
            for field in FIELDS
        },
        "required": FIELDS,
    },
}


def build_system_prompt(hints: dict) -> str:
    return (
        "You are extracting structured information about a biosecurity training program from its webpage content.\n\n"
        "The program is believed to be:\n"
        f"- Name: {hints.get('name', 'unknown')}\n"
        f"- Host organization: {hints.get('lead_org', 'unknown')}\n"
        f"- Country: {hints.get('country', 'unknown')}\n"
        f"- Type: {hints.get('type', 'unknown')}\n\n"
        "Confirm or correct each field based on the page content. "
        "For every field, provide a verbatim snippet from the page as evidence. "
        "If a field is not mentioned on the page, set both value and evidence to empty strings."
    )


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_grounding(evidence: str, raw_text: str) -> bool:
    if not evidence or not evidence.strip():
        return False
    return fuzz.partial_ratio(normalize(evidence), normalize(raw_text)) >= 90


def detect_hint_conflicts(fields: dict, hints: dict) -> list:
    conflicts = []
    for hint_key, field_key in HINT_TO_FIELD.items():
        hint_val = hints.get(hint_key, "")
        extracted_val = fields.get(field_key, {}).get("value", "")
        if not hint_val or not extracted_val:
            continue
        if normalize(hint_val) != normalize(extracted_val):
            conflicts.append({
                "field": field_key,
                "hint_value": hint_val,
                "extracted_value": extracted_val,
                "evidence": fields[field_key].get("evidence", ""),
            })
    return conflicts


async def extract(client: anthropic.AsyncAnthropic, record: dict) -> dict:
    hints = record.get("hints", {})
    raw_text = record.get("raw_text", "")
    system = build_system_prompt(hints)
    user_content = raw_text if raw_text.strip() else "[Page content could not be fetched.]"

    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system,
                tools=[TOOL_DEFINITION],
                tool_choice={"type": "tool", "name": "extract_program_fields"},
                messages=[{"role": "user", "content": user_content}],
            )
            tool_input = next(
                block.input for block in response.content if block.type == "tool_use"
            )
            for field in FIELDS:
                if field not in tool_input:
                    raise ValueError(f"Missing required field: {field}")

            annotated_fields = {}
            for field in FIELDS:
                entry = tool_input[field]
                annotated_fields[field] = {
                    "value": entry.get("value", ""),
                    "evidence": entry.get("evidence", ""),
                    "grounded": check_grounding(entry.get("evidence", ""), raw_text),
                }

            hint_conflicts = detect_hint_conflicts(tool_input, hints)

            return {
                "extraction_status": "ok",
                "fields": annotated_fields,
                "hint_conflicts": hint_conflicts,
                "failure_reason": "",
            }

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    empty_fields = {f: {"value": "", "evidence": "", "grounded": False} for f in FIELDS}
    return {
        "extraction_status": "failed",
        "fields": empty_fields,
        "hint_conflicts": [],
        "failure_reason": last_error,
    }


def build_csv_row(record: dict, result: dict) -> dict:
    row = {
        "url": record["url"],
        "source_doc_id": record.get("source_doc_id", ""),
        "fetch_status": record.get("fetch_status", ""),
        "fetch_method": record.get("fetch_method", ""),
        "fetched_at": record.get("fetched_at", ""),
        "classification_status": record.get("classification_status", ""),
        "classification_confidence": record.get("classification_confidence", ""),
        "classification_reasoning": record.get("classification_reasoning", ""),
        "entity_type": record.get("entity_type", ""),
        "extraction_status": result["extraction_status"],
        "failure_reason": result.get("failure_reason", ""),
        "hint_conflicts": json.dumps(result["hint_conflicts"]) if result["hint_conflicts"] else "",
        "ungrounded_fields": ",".join(
            f for f, v in result["fields"].items() if not v["grounded"]
        ),
    }
    for field in FIELDS:
        row[field] = result["fields"][field]["value"]
    return row


CSV_COLUMNS = (
    ["url", "source_doc_id", "fetch_status", "fetch_method", "fetched_at",
     "classification_status", "classification_confidence", "classification_reasoning",
     "entity_type", "extraction_status", "failure_reason"]
    + FIELDS
    + ["ungrounded_fields", "hint_conflicts"]
)


async def process_file(sem: asyncio.Semaphore, client: anthropic.AsyncAnthropic,
                       path: str, index: int, total: int) -> tuple:
    async with sem:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)

        classification = record.get("classification_status", "")
        if classification == "accept":
            result = await extract(client, record)
            status = result["extraction_status"]
            print(f"[{status}] ({index}/{total}) {record['url']}")
        else:
            # Skip extraction for review/rejected/error records
            empty_fields = {f: {"value": "", "evidence": "", "grounded": False} for f in FIELDS}
            result = {
                "extraction_status": "skipped",
                "fields": empty_fields,
                "hint_conflicts": [],
                "failure_reason": f"classification_status={classification}",
            }
            print(f"[skipped:{classification}] ({index}/{total}) {record['url']}")

        row = build_csv_row(record, result)
        return index, row


async def main():
    classified_files = sorted(glob.glob(os.path.join(CLASSIFIED_DIR, "*.json")))
    if not classified_files:
        print(f"No JSON files found in {CLASSIFIED_DIR}. Run stage2_classify.py first.")
        return

    os.makedirs("output", exist_ok=True)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"Cleared existing {OUTPUT_CSV}")

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Count records by classification status
    accept_count = sum(
        1 for p in classified_files
        if json.load(open(p, encoding="utf-8")).get("classification_status") == "accept"
    )
    print(f"Processing {len(classified_files)} records ({accept_count} accepted, "
          f"{len(classified_files) - accept_count} skipped) (concurrency={CONCURRENCY})")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        process_file(sem, client, path, i, len(classified_files))
        for i, path in enumerate(classified_files, 1)
    ]
    results = await asyncio.gather(*tasks)

    # Sort by original index to preserve input order in the CSV
    results.sort(key=lambda x: x[0])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for _, row in results:
            writer.writerow(row)

    print(f"\nDone. Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
