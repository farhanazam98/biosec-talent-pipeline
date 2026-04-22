import glob
import json
import os
import re
import time

import anthropic
import yaml
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = "data/raw"
CLASSIFIED_DIR = "data/classified"
CONFIG_PATH = "config/classification.yaml"
MODEL = "claude-haiku-4-5-20251001"
POLL_INTERVAL = 30  # seconds between batch status checks

SYSTEM_PROMPT = """\
You are classifying whether a webpage describes a biosecurity talent pipeline entity.

## In Scope

A URL is in scope if it describes ANY of the following:

**Programs** with identifiable individual participants, defined activities, time- or credential-bounded scope, and biosecurity-relevant content. This includes:
- Degrees, certificates, short courses, summer schools
- Fellowships, internships, competitions, scholarships, mentorship cohorts
- Conferences with structured early-career tracks
- Government training programs
- Bilateral/multilateral capacity-building trainings
- Lab network training programs
- Professional association certification programs

**Funders** whose primary or named focus includes biosecurity talent or workforce development (e.g., Open Philanthropy biosecurity grantmaking, Coefficient Giving). A funder qualifies if biosecurity is a named program area, not if biosecurity is one of dozens of incidental grants.

## Out of Scope

- Articles, reports, blog posts, press releases about programs or the field
- National biosecurity strategy documents and white papers
- Generic org "about us" or homepage URLs without a specific program described
- Funders with no named biosecurity focus
- Programs outside biosecurity scope: general public health, general epidemiology, biotech unrelated to dual-use risk, agricultural/plant/livestock biosecurity
- One-off events (single webinars, standalone lectures) without ongoing cohort structure
- Professional associations as entities, unless the URL describes a specific certification or training program the association runs

## Edge Cases

- **Inactive or closed programs** → IN SCOPE. Activity status is a separate field, not a rejection criterion.
- **Degree programs with a biosecurity module or specialization** → IN SCOPE if the biosecurity content is substantive (not a single elective among many).
- **Multi-country programs** (e.g., Erasmus Mundus) → IN SCOPE as a single entity.
- **One Health programs** → IN SCOPE only if they explicitly address deliberate or accidental biological risk (GCBR-relevant framing), not general zoonoses surveillance.

## Confidence Calibration

Your confidence score must be calibrated meaningfully:
- **High confidence (≥ 0.85)**: the page content clearly and unambiguously matches an in-scope or out-of-scope pattern. The decision is obvious.
- **Medium confidence (0.5–0.8)**: borderline cases. Use this range for funders with partial biosecurity focus, programs with ambiguous activity status, pages with thin or unclear content, or anything where a reasonable person might disagree.
- **Low confidence (< 0.5)**: very uncertain. You are guessing.

Examples:
- A fellowship page titled "Emerging Leaders in Biosecurity" with a detailed curriculum → is_pipeline_entity: true, confidence: 0.95
- A university homepage listing 50 departments with no specific biosecurity program described → is_pipeline_entity: false, confidence: 0.92
- A news article about a new biosecurity fellowship → is_pipeline_entity: false, confidence: 0.90
- A One Health program that mentions "antimicrobial resistance" but not deliberate biological threats → is_pipeline_entity: false, confidence: 0.65
- A funder page that lists biosecurity as one of 20 grant areas with no dedicated program → is_pipeline_entity: false, confidence: 0.70
- An inactive fellowship with only a brief archived description remaining → is_pipeline_entity: true, confidence: 0.75

## Hints

The following hints come from a prior (unverified) research pass. Treat them as beliefs to verify, not as ground truth:
- Name: {name}
- Host organization: {lead_org}
- Country: {country}
- Type: {type}
- Active status: {active_status}

Confirm or correct based on the page content.
"""

CLASSIFY_TOOL = {
    "name": "classify_program",
    "description": "Classify whether this URL represents a biosecurity talent pipeline entity.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_pipeline_entity": {
                "type": "boolean",
                "description": "True if the page describes an in-scope biosecurity talent pipeline entity.",
            },
            "entity_type": {
                "type": "string",
                "enum": ["program", "funder", "other"],
                "description": "Type of entity. Use 'program' for training/fellowship/course, 'funder' for funding organizations, 'other' if in scope but neither.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score from 0.0 to 1.0. High only when the page clearly matches an in-scope or out-of-scope pattern.",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the classification decision.",
            },
            "evidence": {
                "type": "string",
                "description": "A verbatim snippet from the page supporting the decision.",
            },
        },
        "required": ["is_pipeline_entity", "entity_type", "confidence", "reasoning", "evidence"],
    },
}


def url_to_filename(url):
    slug = re.sub(r"https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:120] + ".json"


def build_system_prompt(hints):
    return SYSTEM_PROMPT.format(
        name=hints.get("name", "unknown"),
        lead_org=hints.get("lead_org", "unknown"),
        country=hints.get("country", "unknown"),
        type=hints.get("type", "unknown"),
        active_status=hints.get("active_status", "unknown"),
    )


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def route(is_pipeline_entity, confidence, config):
    high_accept = config["high_accept_threshold"]
    high_reject = config["high_reject_threshold"]
    if is_pipeline_entity and confidence >= high_accept:
        return "accept"
    elif not is_pipeline_entity and confidence >= high_reject:
        return "rejected"
    else:
        return "review"


def main():
    config = load_config()

    # Load all Stage 1 records
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not raw_files:
        print(f"No JSON files found in {RAW_DIR}. Run stage1_ingest.py first.")
        return

    records = {}
    for path in raw_files:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        custom_id = url_to_filename(record["url"])
        records[custom_id] = record

    # Clear previous output
    os.makedirs(CLASSIFIED_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(CLASSIFIED_DIR, "*.json"))
    for path in existing:
        os.remove(path)
    if existing:
        print(f"Cleared {len(existing)} existing files from {CLASSIFIED_DIR}/")

    # Build batch requests
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    batch_requests = []
    for custom_id, record in records.items():
        hints = record.get("hints", {})
        raw_text = record.get("raw_text", "")
        user_content = raw_text if raw_text.strip() else "[Page content could not be fetched.]"

        batch_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": 1024,
                "system": [
                    {
                        "type": "text",
                        "text": build_system_prompt(hints),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "tools": [CLASSIFY_TOOL],
                "tool_choice": {"type": "tool", "name": "classify_program"},
                "messages": [{"role": "user", "content": user_content}],
            },
        })

    print(f"Submitting batch of {len(batch_requests)} records...")
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Batch created: {batch.id}")

    # Poll until complete
    while batch.processing_status != "ended":
        time.sleep(POLL_INTERVAL)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(
            f"  Status: {batch.processing_status} | "
            f"succeeded={counts.succeeded} failed={counts.errored} "
            f"processing={counts.processing} pending={counts.pending}"
        )

    print(f"Batch complete. Processing results...")

    # Process results
    accept_count = 0
    review_count = 0
    reject_count = 0
    error_count = 0

    for result in client.messages.batches.results(batch.id):
        custom_id = result.custom_id
        record = records[custom_id]

        if result.result.type == "succeeded":
            message = result.result.message
            try:
                tool_input = next(
                    block.input for block in message.content if block.type == "tool_use"
                )
                is_entity = tool_input["is_pipeline_entity"]
                confidence = tool_input["confidence"]
                status = route(is_entity, confidence, config)

                record["classification_status"] = status
                record["classification_confidence"] = confidence
                record["classification_reasoning"] = tool_input.get("reasoning", "")
                record["classification_evidence"] = tool_input.get("evidence", "")
                record["entity_type"] = tool_input.get("entity_type", "other")

            except Exception as e:
                record["classification_status"] = "error"
                record["classification_confidence"] = 0.0
                record["classification_reasoning"] = f"Parse error: {e}"
                record["classification_evidence"] = ""
                record["entity_type"] = ""
        else:
            error_msg = str(result.result.error) if hasattr(result.result, "error") else "Unknown batch error"
            record["classification_status"] = "error"
            record["classification_confidence"] = 0.0
            record["classification_reasoning"] = error_msg
            record["classification_evidence"] = ""
            record["entity_type"] = ""

        # Write classified record
        out_path = os.path.join(CLASSIFIED_DIR, custom_id)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        status = record["classification_status"]
        if status == "accept":
            accept_count += 1
        elif status == "review":
            review_count += 1
        elif status == "rejected":
            reject_count += 1
        else:
            error_count += 1

    total = accept_count + review_count + reject_count + error_count
    print(f"\nDone. {total} records classified:")
    print(f"  accept:   {accept_count}")
    print(f"  review:   {review_count}")
    print(f"  rejected: {reject_count}")
    print(f"  error:    {error_count}")
    print(f"\nResults written to {CLASSIFIED_DIR}/")


if __name__ == "__main__":
    main()
