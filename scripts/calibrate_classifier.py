"""
Calibration harness for the classification stage.

Runs the classifier against labeled fixtures, sweeps threshold pairs,
and recommends the best (accept_threshold, reject_threshold) combo.

Usage:
    python scripts/calibrate_classifier.py
"""

import asyncio
import csv
import glob
import json
import os
import sys

import anthropic
from dotenv import load_dotenv

# Import classifier internals from stage2_classify
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stage2_classify import (
    CLASSIFY_TOOL,
    MODEL,
    build_system_prompt,
)

load_dotenv()

FIXTURES_DIR = "tests/fixtures/classification"
LABELS_CSV = "docs/labeling_candidates.csv"
REPORT_PATH = "docs/calibration_report.md"
CONFIG_PATH = "config/classification.yaml"
MAX_RETRIES = 3
CONCURRENCY = 3

THRESHOLD_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


async def classify_one(sem, client, fixture, index, total):
    """Run the classifier on a single fixture. Returns (fixture, result)."""
    async with sem:
        hints = fixture.get("hints", {})
        raw_text = fixture.get("raw_text", "")
        system = build_system_prompt(hints)
        user_content = raw_text if raw_text.strip() else "[Page content could not be fetched.]"

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=[
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=[CLASSIFY_TOOL],
                    tool_choice={"type": "tool", "name": "classify_program"},
                    messages=[{"role": "user", "content": user_content}],
                )
                tool_input = next(
                    block.input
                    for block in response.content
                    if block.type == "tool_use"
                )
                result = {
                    "is_pipeline_entity": tool_input["is_pipeline_entity"],
                    "confidence": tool_input["confidence"],
                    "reasoning": tool_input.get("reasoning", ""),
                }
                print(f"  [{index}/{total}] {fixture['url'][:80]} -> entity={result['is_pipeline_entity']} conf={result['confidence']:.2f}")
                return fixture, result

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  [{index}/{total}] {fixture['url'][:80]} -> ERROR: {e}")
                    return fixture, {
                        "is_pipeline_entity": False,
                        "confidence": 0.0,
                        "reasoning": f"Error: {e}",
                    }


def route(is_entity, confidence, accept_thresh, reject_thresh):
    if is_entity and confidence >= accept_thresh:
        return "accept"
    elif not is_entity and confidence >= reject_thresh:
        return "rejected"
    else:
        return "review"


def evaluate_thresholds(results, accept_thresh, reject_thresh):
    """Compute error counts for a given threshold pair."""
    counts = {
        "true_accept": 0,
        "true_reject": 0,
        "true_review": 0,
        "wrong_reject": 0,   # expected accept, got rejected (HIGH COST)
        "wrong_accept": 0,   # expected rejected, got accepted (MEDIUM COST)
        "bumped_to_review": 0,
    }
    for fixture, result in results:
        expected = fixture["expected_label"]
        actual = route(result["is_pipeline_entity"], result["confidence"],
                       accept_thresh, reject_thresh)
        if expected == actual:
            if actual == "accept":
                counts["true_accept"] += 1
            elif actual == "rejected":
                counts["true_reject"] += 1
            else:
                counts["true_review"] += 1
        elif expected == "accept" and actual == "rejected":
            counts["wrong_reject"] += 1
        elif expected == "rejected" and actual == "accept":
            counts["wrong_accept"] += 1
        else:
            counts["bumped_to_review"] += 1
    return counts


def sync_labels_from_csv():
    """Sync expected_label from the labeling CSV into fixture JSON files."""
    if not os.path.exists(LABELS_CSV):
        print(f"No labeling CSV found at {LABELS_CSV}, using existing fixture labels.")
        return

    with open(LABELS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        url_to_label = {}
        for row in reader:
            label = row.get("suggested_label", "").strip()
            url = row.get("url", "").strip()
            if url and label:
                url_to_label[url] = label

    fixture_files = sorted(glob.glob(os.path.join(FIXTURES_DIR, "*.json")))
    updated = 0
    for path in fixture_files:
        with open(path, encoding="utf-8") as f:
            fixture = json.load(f)
        url = fixture.get("url", "")
        if url in url_to_label:
            new_label = url_to_label[url]
            if fixture.get("expected_label") != new_label:
                fixture["expected_label"] = new_label
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(fixture, f, ensure_ascii=False, indent=2)
                updated += 1

    if updated:
        print(f"Synced labels from {LABELS_CSV}: {updated} fixture(s) updated.")
    else:
        print(f"Labels from {LABELS_CSV} already match fixtures.")


async def main():
    # Sync labels from CSV into fixtures
    sync_labels_from_csv()

    # Load fixtures
    fixture_files = sorted(glob.glob(os.path.join(FIXTURES_DIR, "*.json")))
    if not fixture_files:
        print(f"No fixtures found in {FIXTURES_DIR}/")
        return

    fixtures = []
    for path in fixture_files:
        with open(path) as f:
            fixtures.append(json.load(f))

    print(f"Loaded {len(fixtures)} fixtures")
    expected_counts = {}
    for fix in fixtures:
        expected_counts[fix["expected_label"]] = expected_counts.get(fix["expected_label"], 0) + 1
    for label, count in sorted(expected_counts.items()):
        print(f"  {label}: {count}")

    # Run classifier on all fixtures
    print(f"\nRunning classifier (model={MODEL}, concurrency={CONCURRENCY})...")
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        classify_one(sem, client, fix, i, len(fixtures))
        for i, fix in enumerate(fixtures, 1)
    ]
    results = await asyncio.gather(*tasks)

    # Print raw scores
    print("\n--- Raw Classifier Scores ---")
    print(f"{'URL':<80} {'Expected':<10} {'Entity?':<8} {'Conf':<6} {'Reasoning'}")
    print("-" * 140)
    for fixture, result in sorted(results, key=lambda x: x[0]["url"]):
        url = fixture["url"][:78]
        exp = fixture["expected_label"]
        ent = "T" if result["is_pipeline_entity"] else "F"
        conf = f"{result['confidence']:.2f}"
        reason = result["reasoning"][:50]
        print(f"{url:<80} {exp:<10} {ent:<8} {conf:<6} {reason}")

    # Sweep threshold grid
    print("\n--- Threshold Grid ---")
    print(f"{'Accept':<8} {'Reject':<8} {'TrueAcc':<8} {'TrueRej':<8} {'TrueRev':<8} {'WrongRej':<9} {'WrongAcc':<9} {'Bumped':<8} {'Score'}")
    print("-" * 95)

    best_score = -999
    best_pair = (0.85, 0.85)
    best_counts = None
    grid_rows = []

    for at in THRESHOLD_GRID:
        for rt in THRESHOLD_GRID:
            counts = evaluate_thresholds(results, at, rt)
            # Score: minimize wrong_reject (weight 3) and wrong_accept (weight 1)
            # maximize true_accept + true_reject
            score = (
                counts["true_accept"] + counts["true_reject"] + counts["true_review"]
                - 3 * counts["wrong_reject"]
                - 1 * counts["wrong_accept"]
            )
            grid_rows.append((at, rt, counts, score))
            print(
                f"{at:<8.2f} {rt:<8.2f} "
                f"{counts['true_accept']:<8} {counts['true_reject']:<8} {counts['true_review']:<8} "
                f"{counts['wrong_reject']:<9} {counts['wrong_accept']:<9} {counts['bumped_to_review']:<8} "
                f"{score}"
            )
            if score > best_score or (
                score == best_score and counts["wrong_reject"] < best_counts["wrong_reject"]
            ):
                best_score = score
                best_pair = (at, rt)
                best_counts = counts

    print(f"\n--- Recommendation ---")
    print(f"Best thresholds: accept={best_pair[0]:.2f}, reject={best_pair[1]:.2f}")
    print(f"  True accepts:  {best_counts['true_accept']}")
    print(f"  True rejects:  {best_counts['true_reject']}")
    print(f"  True reviews:  {best_counts['true_review']}")
    print(f"  Wrong rejects: {best_counts['wrong_reject']} (HIGH COST)")
    print(f"  Wrong accepts: {best_counts['wrong_accept']} (MEDIUM COST)")
    print(f"  Bumped to review: {best_counts['bumped_to_review']}")

    # Write calibration report
    with open(REPORT_PATH, "w") as f:
        f.write("# Classification Calibration Report\n\n")
        f.write(f"Model: `{MODEL}`\n")
        f.write(f"Fixtures: {len(fixtures)} records ({expected_counts.get('accept',0)} accept, ")
        f.write(f"{expected_counts.get('rejected',0)} rejected, {expected_counts.get('review',0)} review)\n\n")

        f.write("## Fixture Results\n\n")
        f.write("| URL | Expected | is_entity | Confidence | Reasoning |\n")
        f.write("|---|---|---|---|---|\n")
        for fixture, result in sorted(results, key=lambda x: x[0]["url"]):
            url = fixture["url"]
            exp = fixture["expected_label"]
            ent = str(result["is_pipeline_entity"])
            conf = f"{result['confidence']:.2f}"
            reason = result["reasoning"].replace("|", "/").replace("\n", " ")
            f.write(f"| {url} | {exp} | {ent} | {conf} | {reason} |\n")

        f.write("\n## Threshold Grid (top 10 by score)\n\n")
        f.write("Score = true_correct - 3*wrong_rejects - 1*wrong_accepts\n\n")
        f.write("| Accept Thresh | Reject Thresh | True Acc | True Rej | True Rev | Wrong Rej | Wrong Acc | Bumped | Score |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        sorted_grid = sorted(grid_rows, key=lambda x: (-x[3], x[2]["wrong_reject"]))
        for at, rt, counts, score in sorted_grid[:10]:
            f.write(
                f"| {at:.2f} | {rt:.2f} | {counts['true_accept']} | {counts['true_reject']} | "
                f"{counts['true_review']} | {counts['wrong_reject']} | {counts['wrong_accept']} | "
                f"{counts['bumped_to_review']} | {score} |\n"
            )

        f.write(f"\n## Recommended Thresholds\n\n")
        f.write(f"- **high_accept_threshold**: {best_pair[0]:.2f}\n")
        f.write(f"- **high_reject_threshold**: {best_pair[1]:.2f}\n\n")
        f.write(f"**Rationale**: minimizes wrong-rejects (losing real pipeline entries) first, ")
        f.write(f"then wrong-accepts (noise flowing to extraction). ")
        f.write(f"The review tier absorbs borderline cases.\n\n")
        f.write(f"Wrong rejects: {best_counts['wrong_reject']} | ")
        f.write(f"Wrong accepts: {best_counts['wrong_accept']} | ")
        f.write(f"Bumped to review: {best_counts['bumped_to_review']}\n")

    print(f"\nReport written to {REPORT_PATH}")

    with open(CONFIG_PATH, "w") as f:
        f.write(f"high_accept_threshold: {best_pair[0]:.2f}\n")
        f.write(f"high_reject_threshold: {best_pair[1]:.2f}\n")
    print(f"Wrote thresholds to {CONFIG_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
