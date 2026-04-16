import csv
import glob
import json
import os
import re
import time
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()

RAW_DIR = "data/raw"
OUTPUT_CSV = "output/stage2_results.csv"
MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3

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
    "country": "Country or countries where the program is delivered",
    "organisation_funding_course": "Funder(s) of the program",
    "expected_outcomes": "Stated learning or career outcomes",
    "syllabus_course_materials": "Topics, modules, or curriculum links",
    "target_audience": "Career stage, background, or nationality requirements for applicants",
    "financial_support_available": "Stipends, scholarships, travel grants, or fee waivers",
    "visa_travel_constraints": "Nationality restrictions or travel obligations",
    "languages": "Delivery language(s)",
    "year_established": "Year the program was founded or first offered",
    "income_classification": "One of: HIC | LMIC | Both",
    "format": "e.g. in-person, online, hybrid, part-time, full-time",
    "focus_area": "e.g. biosurveillance, policy, lab biosafety, threat assessment",
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


def extract(client: anthropic.Anthropic, record: dict) -> dict:
    hints = record.get("hints", {})
    raw_text = record.get("raw_text", "")
    system = build_system_prompt(hints)
    user_content = raw_text if raw_text.strip() else "[Page content could not be fetched.]"

    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
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
            # Validate all required fields are present
            for field in FIELDS:
                if field not in tool_input:
                    raise ValueError(f"Missing required field: {field}")

            # Grounding check
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
                time.sleep(2 ** attempt)

    # All retries exhausted
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
     "extraction_status", "failure_reason", "hint_conflicts", "ungrounded_fields"]
    + FIELDS
)


def main():
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not raw_files:
        print(f"No JSON files found in {RAW_DIR}. Run stage1_ingest.py first.")
        return

    os.makedirs("output", exist_ok=True)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"Processing {len(raw_files)} records")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, path in enumerate(raw_files, 1):
            with open(path, encoding="utf-8") as jf:
                record = json.load(jf)

            url = record["url"]
            result = extract(client, record)
            row = build_csv_row(record, result)
            writer.writerow(row)
            f.flush()

            status = result["extraction_status"]
            print(f"[{status}] ({i}/{len(raw_files)}) {url}")

    print(f"\nDone. Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
