import csv
import glob
import json
import os
import re
import time

import anthropic
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()

CLASSIFIED_DIR = "data/classified"
OUTPUT_CSV = "output/stage3_results.csv"
MODEL = "claude-sonnet-4-6"
POLL_INTERVAL = 30  # seconds between batch status checks

PIPELINE_TYPE_TO_CATEGORY = {
    "degree": "formal_training",
    "certificate": "formal_training",
    "short_course": "formal_training",
    "summer_school": "formal_training",
    "online": "formal_training",
    "fellowship": "non_degree_structured",
    "internship": "non_degree_structured",
    "competition": "non_degree_structured",
    "scholarship": "non_degree_structured",
    "mentorship": "non_degree_structured",
    "conference": "non_degree_structured",
    "association": "non_degree_structured",
    "gov_training": "gov_institutional",
    "bilateral": "gov_institutional",
    "multilateral": "gov_institutional",
    "regional_body": "gov_institutional",
    "lab_network": "gov_institutional",
    "funder_initiative": "gov_institutional",
    "national_strategy": "gov_institutional",
    "other": "",
}

PIPELINE_TYPES = list(PIPELINE_TYPE_TO_CATEGORY.keys())

FIELDS = [
    "name_and_title",
    "organisation_providing_course",
    "pipeline_type",
    "country",
    "organisation_funding_course",
    "expected_outcomes",
    "syllabus_course_materials",
    "career_stage",
    "financial_support_available",
    "visa_travel_constraints",
    "languages",
    "year_established",
    "active_status",
    "income_classification",
    "format",
    "focus_area",
    "ai_risks_content_included",
    "dual_use_risks_content_included",
]

FIELD_DESCRIPTIONS = {
    "name_and_title": "Full official program name",
    "organisation_providing_course": "Host / delivering organization",
    "pipeline_type": (
        "Exactly one of these values:\n"
        "FORMAL TRAINING CATEGORY:\n"
        "  degree — full degree program (BSc, MSc, PhD, MPH) with biosecurity content\n"
        "  certificate — non-degree credential program (professional certificate, diploma)\n"
        "  short_course — bounded training (days to weeks), instructor-led, not degree-bearing\n"
        "  summer_school — intensive residential program, typically annual, aimed at students/early-career\n"
        "  online — self-paced, open-access, or MOOC-style course not fitting above\n"
        "NON-DEGREE STRUCTURED OPPORTUNITIES CATEGORY:\n"
        "  fellowship — funded position with structured activities and defined cohort\n"
        "  internship — time-bounded placement at a host organization\n"
        "  competition — competitive event with biosecurity-relevant challenges\n"
        "  scholarship — financial award tied to study/research in biosecurity\n"
        "  mentorship — structured mentor-mentee pairing with defined program activities\n"
        "  conference — conference with a structured early-career track (not general attendance)\n"
        "  association — specific program or certification run by a professional association\n"
        "GOVERNMENT & INSTITUTIONAL CATEGORY:\n"
        "  gov_training — government-run domestic training program\n"
        "  bilateral — two-country capacity-building partnership\n"
        "  multilateral — multi-country or international org training initiative\n"
        "  regional_body — program run by a regional body (e.g., Africa CDC, EU agency)\n"
        "  lab_network — training delivered through a laboratory network\n"
        "  funder_initiative — program run or funded by a dedicated biosecurity funder\n"
        "  national_strategy — training component of a national strategy document (rare)\n"
        "  other — use only if nothing above fits"
    ),
    "country": (
        "Pipe-delimited list of full country names where the program is delivered (e.g. USA|Canada). "
        "Use 'Global' if not specific to any country or if 40+ countries are covered. "
        "Use 'Regional – <region>' (e.g. 'Regional – Africa') if the program targets a broad region without listing specific countries."
    ),
    "organisation_funding_course": "Funder(s) of the program",
    "expected_outcomes": "Stated learning or career outcomes",
    "syllabus_course_materials": "Topics, modules, or curriculum links",
    "career_stage": (
        "Pipe-delimited list from this vocabulary: "
        "undergraduate | postgraduate | early_career | mid_career | senior | professional | unknown. "
        "undergraduate = current BSc/BA students. "
        "postgraduate = current MSc/PhD students or recent graduates. "
        "early_career = 0-5 years post-degree professionals. "
        "mid_career = 5-15 years professional experience. "
        "senior = 15+ years or leadership/director level. "
        "professional = practicing professionals (lab staff, clinicians, officials) regardless of seniority. "
        "unknown = not mentioned on the page. "
        "Choose all that apply."
    ),
    "financial_support_available": (
        "Classify as exactly one of: full | partial | free | none | unknown. "
        "full = tuition, stipend, and/or living costs covered. "
        "partial = some costs covered (travel grant, fee waiver, accommodation only, etc.). "
        "free = no tuition fee but no additional financial support. "
        "none = explicitly no financial support available. "
        "unknown = not mentioned on the page."
    ),
    "visa_travel_constraints": (
        "Classify as exactly one of: yes | no | n/a | unknown. "
        "yes = any visa requirement, travel obligation, or nationality restriction is mentioned. "
        "no = page explicitly states no constraints. "
        "n/a = not applicable (e.g. fully online courses with no travel component). "
        "unknown = not mentioned on the page."
    ),
    "languages": (
        "Pipe-delimited list of delivery language names in English (e.g. English|French|Spanish). "
        "Omit parenthetical notes such as '(simultaneous interpretation)'."
    ),
    "year_established": "Year the program was founded or first offered",
    "active_status": (
        "Classify as exactly one of: active | inactive | unknown. "
        "active = page indicates the program is currently running, has a recent or upcoming cohort, "
        "open application window, or recent activity (within ~2 years of the page's most recent content). "
        "inactive = page explicitly states the program has concluded, been discontinued, is no longer accepting applications, "
        "or shows clear signals of being defunct (e.g. 'final cohort 2018', archived project page). "
        "unknown = no clear signal either way."
    ),
    "income_classification": "One of: HIC | LMIC | Both",
    "format": "e.g. in-person, online, hybrid, part-time, full-time",
    "focus_area": (
        "Pipe-delimited list of tags from this controlled vocabulary: "
        "dna_synthesis_screening (nucleic acid synthesis screening, sequence screening) | "
        "policy_governance (biosecurity governance, arms control, BWC, regulatory frameworks) | "
        "biosurveillance (disease surveillance, early warning, outbreak detection) | "
        "lab_biosafety (lab biosafety, biorisk management, containment, lab techniques) | "
        "pandemic_preparedness_and_response (pandemic response, epidemic preparedness, health security) | "
        "ai_biosecurity (AI risks in biology, AI biosecurity governance). "
        "Choose only tags that clearly apply. Use a short free-text value only if nothing fits."
    ),
    "ai_risks_content_included": "Y or N — does the program cover AI risks?",
    "dual_use_risks_content_included": "Y or N — does the program cover dual-use risks?",
}

# Hints that map directly to extracted fields for conflict detection
# "type" hint is a category-level value (e.g. "formal_training") compared
# against pipeline_category derived from the fine-grained pipeline_type
HINT_TO_FIELD = {
    "name": "name_and_title",
    "lead_org": "organisation_providing_course",
    "country": "country",
    "type": "pipeline_category",
    "active_status": "active_status",
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


def url_to_filename(url):
    slug = re.sub(r"https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:120] + ".json"


def build_system_prompt(hints: dict) -> str:
    return (
        "You are extracting structured information about a biosecurity training program from its webpage content.\n\n"
        "The program is believed to be:\n"
        f"- Name: {hints.get('name', 'unknown')}\n"
        f"- Host organization: {hints.get('lead_org', 'unknown')}\n"
        f"- Country: {hints.get('country', 'unknown')}\n"
        f"- Category: {hints.get('type', 'unknown')}\n\n"
        "Confirm or correct each field based on the page content. "
        "For every field, provide a verbatim snippet from the page as evidence. "
        "If a field is not mentioned on the page, set both value and evidence to empty strings.\n\n"
        "## Pipeline Type\n\n"
        "The category hint above is a broad grouping from a prior research pass. "
        "For the pipeline_type field, extract the specific type from the page content — "
        "do not inherit the category hint.\n\n"
        "Examples:\n"
        "- An MSc in Biosecurity → pipeline_type: degree\n"
        "- A 5-day WHO workshop → pipeline_type: short_course\n"
        "- ELBI Fellowship with annual cohort → pipeline_type: fellowship\n"
        "- German Biosecurity Programme (bilateral capacity building) → pipeline_type: bilateral\n"
        "- IFBA mentorship pairing program → pipeline_type: mentorship\n"
        "- Africa CDC certification for biosafety professionals → pipeline_type: association\n"
        "- Open Philanthropy biosecurity grants → pipeline_type: funder_initiative"
    )


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


PIPE_DELIMITED_FIELDS = {"focus_area", "career_stage", "country", "languages"}


def normalize_pipe_field(value: str) -> str:
    if not value:
        return value
    items = [re.sub(r"\s*\(.*?\)", "", item).strip() for item in value.split("|")]
    return "|".join(item for item in items if item)


def check_grounding(evidence: str, raw_text: str) -> bool:
    if not evidence or not evidence.strip():
        return False
    return fuzz.partial_ratio(normalize(evidence), normalize(raw_text)) >= 90


def detect_hint_conflicts(fields: dict, hints: dict) -> list:
    # Derive pipeline_category from extracted pipeline_type for comparison
    pipeline_type_val = fields.get("pipeline_type", {}).get("value", "")
    derived_category = PIPELINE_TYPE_TO_CATEGORY.get(pipeline_type_val, "")

    conflicts = []
    for hint_key, field_key in HINT_TO_FIELD.items():
        hint_val = hints.get(hint_key, "")
        if field_key == "pipeline_category":
            extracted_val = derived_category
            evidence = fields.get("pipeline_type", {}).get("evidence", "")
        else:
            extracted_val = fields.get(field_key, {}).get("value", "")
            evidence = fields.get(field_key, {}).get("evidence", "")
        if not hint_val or not extracted_val:
            continue
        if normalize(hint_val) != normalize(extracted_val):
            conflicts.append({
                "field": field_key,
                "hint_value": hint_val,
                "extracted_value": extracted_val,
                "evidence": evidence,
            })
    return conflicts


def process_extraction_result(tool_input: dict, record: dict) -> dict:
    """Process a successful extraction tool output into annotated fields."""
    raw_text = record.get("raw_text", "")
    hints = record.get("hints", {})

    annotated_fields = {}
    for field in FIELDS:
        entry = tool_input[field]
        if isinstance(entry, str):
            entry = {"value": entry, "evidence": ""}
        val = entry.get("value", "")
        if field in PIPE_DELIMITED_FIELDS:
            val = normalize_pipe_field(val)
        annotated_fields[field] = {
            "value": val,
            "evidence": entry.get("evidence", ""),
            "grounded": check_grounding(entry.get("evidence", ""), raw_text),
        }

    hint_conflicts = detect_hint_conflicts(annotated_fields, hints)

    return {
        "extraction_status": "ok",
        "fields": annotated_fields,
        "hint_conflicts": hint_conflicts,
        "failure_reason": "",
    }


def make_empty_result(failure_reason: str) -> dict:
    empty_fields = {f: {"value": "", "evidence": "", "grounded": False} for f in FIELDS}
    return {
        "extraction_status": "failed" if "error" in failure_reason.lower() else "skipped",
        "fields": empty_fields,
        "hint_conflicts": [],
        "failure_reason": failure_reason,
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
        "extraction_status": result["extraction_status"],
        "failure_reason": result.get("failure_reason", ""),
        "hint_conflicts": json.dumps(result["hint_conflicts"]) if result["hint_conflicts"] else "",
        "ungrounded_fields": ",".join(
            f for f, v in result["fields"].items() if not v["grounded"]
        ),
    }
    for field in FIELDS:
        row[field] = result["fields"][field]["value"]
    # Backfill from hints for records that were never extracted
    hints = record.get("hints", {})
    if not row.get("name_and_title"):
        row["name_and_title"] = hints.get("name", "")
    if not row.get("organisation_providing_course"):
        row["organisation_providing_course"] = hints.get("lead_org", "")
    if not row.get("country"):
        row["country"] = hints.get("country", "")
    if not row.get("active_status"):
        row["active_status"] = hints.get("active_status", "")
    # Derive pipeline_category from extracted pipeline_type
    pipeline_type = row.get("pipeline_type", "")
    row["pipeline_category"] = PIPELINE_TYPE_TO_CATEGORY.get(pipeline_type, "")
    return row


# Build CSV columns with pipeline_category inserted after pipeline_type
_FIELDS_WITH_CATEGORY = []
for f in FIELDS:
    _FIELDS_WITH_CATEGORY.append(f)
    if f == "pipeline_type":
        _FIELDS_WITH_CATEGORY.append("pipeline_category")

CSV_COLUMNS = (
    ["url", "source_doc_id", "fetch_status", "fetch_method", "fetched_at",
     "classification_status", "classification_confidence", "classification_reasoning",
     "extraction_status", "failure_reason"]
    + _FIELDS_WITH_CATEGORY
    + ["ungrounded_fields", "hint_conflicts"]
)


def main():
    classified_files = sorted(glob.glob(os.path.join(CLASSIFIED_DIR, "*.json")))
    if not classified_files:
        print(f"No JSON files found in {CLASSIFIED_DIR}. Run stage2_classify.py first.")
        return

    os.makedirs("output", exist_ok=True)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"Cleared existing {OUTPUT_CSV}")

    # Load all records, separate accepted from non-accepted
    all_records = {}  # custom_id -> record
    accepted = {}     # custom_id -> record (only accepted ones)
    for path in classified_files:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        filename = url_to_filename(record["url"])
        custom_id = filename.replace(".json", "")[:64]
        all_records[custom_id] = record
        if record.get("classification_status") == "accept":
            accepted[custom_id] = record

    print(f"Loaded {len(all_records)} records ({len(accepted)} accepted for extraction, "
          f"{len(all_records) - len(accepted)} skipped)")

    # Build batch requests for accepted records
    results = {}  # custom_id -> extraction result

    if accepted:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        batch_requests = []
        for custom_id, record in accepted.items():
            hints = record.get("hints", {})
            raw_text = record.get("raw_text", "")
            user_content = raw_text if raw_text.strip() else "[Page content could not be fetched.]"

            batch_requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": 4096,
                    "system": [
                        {
                            "type": "text",
                            "text": build_system_prompt(hints),
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "tools": [TOOL_DEFINITION],
                    "tool_choice": {"type": "tool", "name": "extract_program_fields"},
                    "messages": [{"role": "user", "content": user_content}],
                },
            })

        print(f"Submitting extraction batch of {len(batch_requests)} records...")
        batch = client.messages.batches.create(requests=batch_requests)
        print(f"Batch created: {batch.id}")

        # Poll until complete
        while batch.processing_status != "ended":
            time.sleep(POLL_INTERVAL)
            batch = client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            print(f"  Status: {batch.processing_status} | {counts}")

        print("Batch complete. Processing results...")

        # Process batch results
        ok_count = 0
        fail_count = 0
        for result in client.messages.batches.results(batch.id):
            custom_id = result.custom_id
            record = accepted[custom_id]

            if result.result.type == "succeeded":
                message = result.result.message
                try:
                    tool_input = next(
                        block.input for block in message.content if block.type == "tool_use"
                    )
                    for field in FIELDS:
                        if field not in tool_input:
                            raise ValueError(f"Missing required field: {field}")

                    results[custom_id] = process_extraction_result(tool_input, record)
                    ok_count += 1
                except Exception as e:
                    results[custom_id] = make_empty_result(f"Parse error: {e}")
                    fail_count += 1
            else:
                error_msg = str(result.result.error) if hasattr(result.result, "error") else "Unknown batch error"
                results[custom_id] = make_empty_result(f"Batch error: {error_msg}")
                fail_count += 1

        print(f"Extraction results: {ok_count} ok, {fail_count} failed")

    # Build CSV rows for all records (accepted + non-accepted)
    rows = []
    for custom_id in all_records:
        record = all_records[custom_id]
        if custom_id in results:
            result = results[custom_id]
        else:
            classification = record.get("classification_status", "")
            result = make_empty_result(f"classification_status={classification}")
        rows.append(build_csv_row(record, result))

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        # Append summary below last entry
        summary_rows = [
            ("Total records", len(rows)),
            ("Successful fetch", sum(1 for r in rows if r.get("fetch_status") == "ok")),
            ("Passes classification", sum(1 for r in rows if r.get("classification_status") == "accept")),
            ("Successful extraction", sum(1 for r in rows if r.get("extraction_status") == "ok")),
        ]

        blank = {c: "" for c in CSV_COLUMNS}
        writer.writerow(blank)
        for label, value in summary_rows:
            summary = {c: "" for c in CSV_COLUMNS}
            summary["url"] = label
            summary["source_doc_id"] = str(value)
            writer.writerow(summary)

    print(f"\nDone. Results written to {OUTPUT_CSV}")
    print(f"\n--- Pipeline Summary ---")
    for label, value in summary_rows:
        print(f"  {label}: {value}")


if __name__ == "__main__":
    main()
