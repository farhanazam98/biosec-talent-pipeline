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
CONCURRENCY = 1  # sequential to stay within 30k input tokens/min rate limit
REQUEST_DELAY = 3  # seconds between API calls

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
    "income_classification": "One of: HIC | LMIC | Both",
    "format": "e.g. in-person, online, hybrid, part-time, full-time",
    "focus_area": (
        "Pipe-delimited list of tags from this controlled vocabulary: "
        "dna_synthesis_screening (nucleic acid synthesis screening, sequence screening) | "
        "policy_governance (biosecurity governance, arms control, BWC, regulatory frameworks) | "
        "biosurveillance (disease surveillance, early warning, outbreak detection) | "
        "lab_biosafety (lab biosafety, biorisk management, containment, lab techniques) | "
        "pandemic_preparedness_and_response (pandemic response, epidemic preparedness, health security) | "
        "one_health (One Health, human-animal-environment interface) | "
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

        except anthropic.RateLimitError as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
                print(f"  Rate limited. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                await asyncio.sleep(wait)
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
            await asyncio.sleep(REQUEST_DELAY)
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
