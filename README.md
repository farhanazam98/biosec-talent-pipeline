# Biosecurity Talent Pipeline

A Python pipeline to catalog global biosecurity fellowship and training programs into a structured database. Built as part of the SPAR research project [*Mapping Global Talent Pipelines in Biosecurity*](https://sparai.org/attachments/proposals/recKAplTbC4arg03R/spar-proposal-1-global-talent-mapping-biosecurity-sengupta-a..pdf).

## What it does

Reads a CSV work queue of program URLs, fetches page content, classifies each page as in-scope or out-of-scope, and uses the Claude API to extract structured metadata against a 17-field schema — including target audience, financial support, geographic focus, and whether the program covers AI or dual-use risks.

See [`docs/design.md`](docs/design.md) for the full architecture.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # then add your Anthropic API key
```

## Usage

```bash
# Stage 1: fetch page content for all URLs in data/work_queue.csv
python src/stage1_ingest.py

# Stage 2: classify records (accept/review/rejected)
python src/stage2_classify.py

# Stage 3: extract structured fields for accepted records
python src/stage3_extract.py
```

Results are written to `output/stage3_results.csv`.

## Project structure

```
data/
  work_queue.csv       # input: URLs + hint metadata
  raw/                 # Stage 1 output: one JSON per program (gitignored)
  classified/          # Stage 2 output: classified JSON (gitignored)
config/
  classification.yaml  # classifier threshold config
docs/
  design.md            # full pipeline design
  calibration_report.md # classifier calibration results
output/                # Stage 3 results (gitignored)
scripts/
  calibrate_classifier.py  # threshold calibration harness
src/
  stage1_ingest.py
  stage2_classify.py
  stage3_extract.py
tests/
  fixtures/classification/  # labeled test data for calibration
```
