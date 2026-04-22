# Biosecurity Talent Pipeline

A Python pipeline to catalog global biosecurity fellowship and training programs into a structured database. Built as part of the SPAR research project [*Mapping Global Talent Pipelines in Biosecurity*](https://sparai.org/attachments/proposals/recKAplTbC4arg03R/spar-proposal-1-global-talent-mapping-biosecurity-sengupta-a..pdf).

## What it does

Reads a CSV work queue of program URLs, fetches page content, and uses the Claude API to extract structured metadata against a 17-field schema — including target audience, financial support, geographic focus, and whether the program covers AI or dual-use risks.

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

# Stage 2: extract structured fields via Claude API
python src/stage2_extract.py
```

Results are written to `output/stage2_results.csv`.

## Project structure

```
data/
  work_queue.csv       # input: URLs + hint metadata
  raw/                 # Stage 1 output: one JSON per program (gitignored)
docs/
  design.md            # full pipeline design
output/                # Stage 2 results (gitignored)
src/
  stage1_ingest.py
  stage2_extract.py
```
