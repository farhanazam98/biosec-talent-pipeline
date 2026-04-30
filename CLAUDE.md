# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This pipeline catalogs global biosecurity fellowship and training programs into a structured database, as part of the SPAR research project *Mapping Global Talent Pipelines in Biosecurity*. It reads a CSV work queue of program URLs, fetches page content, and uses the Claude API to extract structured metadata against a 17-field schema. Full design is in [`docs/design.md`](docs/design.md).

## Setup

```bash
pip install -r requirements.txt
playwright install chromium   # for JS-rendered page fallback
cp env.example .env           # add ANTHROPIC_API_KEY
```

## Running the pipeline

```bash
# Stage 1: fetch page content → data/raw/*.json
python src/stage1_ingest.py

# Stage 2: classify records (Batch API) → data/classified/*.json
python src/stage2_classify.py

# Stage 3: extract structured fields for accepted records → output/stage3_results.csv
python src/stage3_extract.py

# Stage 4: dedup → output/stage4_results.csv
python src/stage4_dedup.py

# Calibration (run before first pipeline use to tune thresholds)
python scripts/calibrate_classifier.py
python scripts/calibrate_dedup.py   # after labeling tests/fixtures/duplicate_candidates.csv
```

## Data flow

```
data/work_queue.csv → [Stage 1] → data/raw/*.json → [Stage 2] → data/classified/*.json → [Stage 3] → output/stage3_results.csv → [Stage 4] → output/stage4_results.csv
```

`data/raw/`, `data/classified/`, and `output/` are generated artifacts — all are gitignored and must never be committed. `.env` contains the Anthropic API key and is also gitignored.

## Architecture

Stages are decoupled by file artifacts so each can be re-run independently. Persisting raw page content in Stage 1 means later stages can be re-run as the schema evolves without re-fetching pages.

### Stage 1 — Ingest (`src/stage1_ingest.py`)

Reads `data/work_queue.csv`. For each row: fetches page content via `trafilatura`, with a Playwright fallback for JS-rendered pages. Writes one JSON file to `data/raw/` per record, including `fetch_method`, `fetched_at`, and `fetch_status` (`ok` | `failed` | `partial`).

Failed fetches still produce a JSON and flow through — failures are never silently dropped.

Work queue columns: `url`, `name_hint`, `lead_org_hint`, `country_hint`, `type_hint`, `active_status_hint`, `region_hint`, `source_doc_id`. All hint columns are nested under `hints{}` in the Stage 1 JSON output to mark the provenance boundary — everything inside `hints` is unverified Gemini output; everything outside is pipeline-produced.

### Stage 2 — Classify (`src/stage2_classify.py`)

Filters non-biosecurity URLs before extraction. Submits all Stage 1 records as a single Anthropic Batch API request using Sonnet 4.6 (`claude-sonnet-4-6`). Classifies each page as a pipeline entity or not, with tiered routing:

- **accept**: `is_pipeline_entity == true` AND `confidence >= 0.85` → extracted in Stage 3
- **rejected**: `is_pipeline_entity == false` AND `confidence >= 0.95` → skipped in Stage 3
- **review**: everything else → skipped in Stage 3, flagged for human review

Thresholds are in `config/classification.yaml`, tuned via `scripts/calibrate_classifier.py` against labeled fixtures in `tests/fixtures/classification/`. Calibration report: `docs/calibration_report.md`.

### Stage 3 — Extract (`src/stage3_extract.py`)

Reads Stage 2 classified JSON files. Only calls Claude API for records with `classification_status == "accept"`. Non-accepted records pass through with empty extraction fields.

Key implementation rules:
- **Model**: pinned to `claude-sonnet-4-6`, never `"latest"`
- **Hints as beliefs**: hints are passed in the system prompt as beliefs to verify, not answers — e.g. *"The program is believed to be called X. Confirm or correct each field based on the page content."* Disagreements are logged in `hint_conflicts`
- **Evidence grounding**: each field's `evidence` snippet is fuzzy-matched against `raw_text` using `rapidfuzz.partial_ratio >= 90`, after normalizing (lowercase, collapse whitespace, strip punctuation). Evidence snippets must be verbatim from `raw_text`. Fields that fail grounding are retained but marked `grounded: false`
- **Retries**: up to 3 times with exponential backoff on parse errors, missing required fields, or API errors. After all retries fail, emit a record with `extraction_status: "failed"` and a populated `failure_reason` — it still flows downstream

### Stage 4 — Dedup (`src/stage4_dedup.py`)

Reads `output/stage3_results.csv`, drops duplicate rows, writes `output/stage4_results.csv`. Two-pass:

- **Pass 1 (rapidfuzz, no API cost)**: pairwise `fuzz.WRatio` over normalized `name_and_title` and `organisation_providing_course`. Pairs with `name_score >= 92 AND org_score >= 80` are confirmed dupes; pairs in `name_score in [55, 92)` OR (`name_score < 55 AND org_score >= 90 AND same country`) are escalated to Pass 2.
- **Pass 2 (Claude Batch API)**: borderline pairs sent to Sonnet 4.6 with forced tool use, returning `same_program: bool, confidence: float`. Pairs marked same with confidence ≥ 0.85 join the confirmed-dupe set. The Pass 1 borderline triggers are deliberately generous so that acronym ↔ full-name pairs (e.g., "ELBI" vs "Emerging Leaders in Biosecurity Initiative") reach Claude.

Confirmed-dupe pairs are merged via union-find. For each cluster the canonical row is picked by ranking on `extraction_status == "ok"` → `fetch_status == "ok"` → most non-empty fields → first occurrence. The output adds three columns: `program_id` (sha256 prefix of normalized canonical name+org), `source_doc_ids` (pipe-joined union of all cluster members), `duplicate_count`.

Thresholds in `config/dedup.yaml`. Calibrate via `scripts/calibrate_dedup.py` against `tests/fixtures/duplicate_candidates.csv` (generated by `python src/stage4_dedup.py --dump-candidates` and labeled by hand).

## Key design decisions

**Do not change without understanding these:**

- **`hints{}` boundary**: hint columns from the work queue are unverified Gemini output. Nesting them under `hints` in Stage 1 JSON marks this clearly. Do not promote hint values to top-level fields or treat them as ground truth.
- **17-field schema is fixed**: Stage 3 uses forced tool use against a schema defined in `docs/design.md`. Do not add, remove, or rename fields without updating the schema table there first. `pipeline_category` is derived programmatically from `pipeline_type`, not extracted.
- **Evidence must be verbatim**: grounding verification uses fuzzy string match against `raw_text`. Paraphrased evidence will fail the check.
- **Models are pinned**: both extraction and classification use `claude-sonnet-4-6`. Never use floating aliases.

## Extraction schema (17 extracted fields + 1 derived)

| # | Field | Notes |
|---|---|---|
| 1 | Name & Title | Full official program name |
| 2 | Organisation Providing Course | Host / delivering organization |
| 3 | Pipeline Type | `degree` \| `certificate` \| `short_course` \| `summer_school` \| `online` \| `fellowship` \| `internship` \| `competition` \| `scholarship` \| `mentorship` \| `conference` \| `association` \| `gov_training` \| `bilateral` \| `multilateral` \| `regional_body` \| `lab_network` \| `funder_initiative` \| `national_strategy` \| `other` |
| 3b | Pipeline Category | Derived from Pipeline Type: `formal_training` \| `non_degree_structured` \| `gov_institutional` |
| 4 | Country | Country/countries where delivered |
| 5 | Organisation Funding Course | Funder(s) |
| 6 | Expected Outcomes | Stated learning or career outcomes |
| 7 | Syllabus / Course Materials | Topics, modules, curriculum links |
| 8 | Career Stage | `undergraduate \| postgraduate \| early_career \| mid_career \| senior \| professional \| unknown` (pipe-delimited) |
| 9 | Financial Support Available | `full \| partial \| free \| none \| unknown` |
| 10 | Visa / Travel Constraints | `yes \| no \| n/a \| unknown` |
| 11 | Language(s) | Delivery language(s) |
| 12 | Year Established | Year founded or first offered |
| 13 | Income Classification | `HIC` \| `LMIC` \| `Both` |
| 14 | Format | e.g. in-person, online, hybrid, part-time, full-time |
| 15 | Focus Area | `dna_synthesis_screening \| policy_governance \| biosurveillance \| lab_biosafety \| pandemic_preparedness_and_response \| ai_biosecurity` (pipe-delimited) |
| 16 | AI Risks Content Included | `Y` \| `N` |
| 17 | Dual-Use Risks Content Included | `Y` \| `N` |

## Code style

Prefer simple, direct implementations. Do not add abstraction layers, base classes, or helper utilities unless explicitly asked. When in doubt, write less code.

## Known limitations and future work

- **Multiple work queue files**: Stage 1 currently reads a single `data/work_queue.csv`. Work queues will eventually follow per-region/per-category naming conventions (e.g. `europe_gov_multilateral.csv`). Stage 1 needs to be updated to glob all CSVs from `data/` and process them, with concurrency/batch processing to handle the larger volume.
- **Single-page extraction**: Stage 2 extracts information from a single fetched page per program. Some programs spread information across multiple pages (e.g. separate curriculum, funding, or application pages). Multi-page crawling is out of scope for the current implementation.

## Open design questions (from `docs/design.md`)

- **Active-status heuristics**: combining `active_status_hint` with fetch signals (e.g. HTTP 404 → `inactive`) needs refinement once real failure patterns are observed.
