import argparse
import csv
import hashlib
import os
import re
import sys
import time

import anthropic
import yaml
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

load_dotenv()

INPUT_CSV = "output/stage3_results.csv"
OUTPUT_CSV = "output/stage4_results.csv"
CANDIDATES_CSV = "tests/fixtures/duplicate_candidates.csv"
CONFIG_PATH = "config/dedup.yaml"
MODEL = "claude-sonnet-4-6"
POLL_INTERVAL = 30

DEFAULTS = {
    "heuristic_name_threshold": 92,
    "heuristic_org_threshold": 80,
    "borderline_name_lower": 80,
    "borderline_org_strong": 95,
    "claude_confidence_threshold": 0.85,
}

JUDGE_TOOL = {
    "name": "judge_duplicate",
    "description": "Decide whether two pipeline entries refer to the same biosecurity program.",
    "input_schema": {
        "type": "object",
        "properties": {
            "same_program": {
                "type": "boolean",
                "description": "True if both entries describe the same program offering by the same host organization.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score from 0.0 to 1.0.",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the decision.",
            },
        },
        "required": ["same_program", "confidence", "reasoning"],
    },
}


def load_config():
    config = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)
    return config


def normalize_for_dedup(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_stage3_csv(path: str):
    """Returns (data_rows, fieldnames). Skips trailing summary block."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    # Stage 3 appends a blank row + summary rows after the data. The summary
    # rows have non-URL labels in the "url" column ("Total records", etc.).
    # Detect data rows by url starting with "http" OR being empty (still a real
    # record even if URL is missing).
    data_rows = []
    for row in all_rows:
        url = (row.get("url") or "").strip()
        if url.lower().startswith("http"):
            data_rows.append(row)
        elif not url and any(row.get(f) for f in fieldnames):
            # Fully empty row → end of data block (separator before summary)
            if not any(row.get(f) for f in fieldnames):
                continue
            data_rows.append(row)
    return data_rows, fieldnames


def country_set(raw: str) -> set:
    """Parse a pipe-delimited country field into a normalized set. Empty for Global/unknown."""
    if not raw:
        return set()
    parts = {normalize_for_dedup(p) for p in raw.split("|") if p.strip()}
    parts = {p for p in parts if p}
    if "global" in parts:
        return set()  # treat Global as wildcard
    return parts


def countries_compatible(set_a: set, set_b: set) -> bool:
    """Soft country check: pass if either side is unknown/Global, or sets overlap."""
    if not set_a or not set_b:
        return True
    return bool(set_a & set_b)


def compute_pair_buckets(rows, config):
    """Run Pass 1 on all pairs. Returns (confirmed, borderline, all_scores)."""
    n = len(rows)
    names = [normalize_for_dedup(r.get("name_and_title", "")) for r in rows]
    orgs = [normalize_for_dedup(r.get("organisation_providing_course", "")) for r in rows]
    country_sets = [country_set(r.get("country", "")) for r in rows]

    # Mark which rows have a usable signature
    has_sig = [bool(names[i]) for i in range(n)]
    eligible_idx = [i for i in range(n) if has_sig[i]]
    if not eligible_idx:
        return [], [], {}

    eligible_names = [names[i] for i in eligible_idx]
    eligible_orgs = [orgs[i] for i in eligible_idx]

    # Pairwise score matrices over eligible rows
    name_matrix = process.cdist(eligible_names, eligible_names, scorer=fuzz.WRatio)
    org_matrix = process.cdist(eligible_orgs, eligible_orgs, scorer=fuzz.WRatio)

    confirmed = []
    borderline = []
    all_scores = {}

    name_hi = config["heuristic_name_threshold"]
    org_hi = config["heuristic_org_threshold"]
    name_lo = config["borderline_name_lower"]
    org_strong = config["borderline_org_strong"]

    for ai, i in enumerate(eligible_idx):
        for aj, j in enumerate(eligible_idx):
            if j <= i:
                continue
            ns = float(name_matrix[ai][aj])
            os_ = float(org_matrix[ai][aj])
            all_scores[(i, j)] = (ns, os_)

            if ns >= name_hi and os_ >= org_hi:
                confirmed.append((i, j))
            elif name_lo <= ns < name_hi and os_ >= org_hi:
                # Name-similar borderline: require strong org match too, otherwise
                # generic shared words (e.g. "Master", "Health") inflate WRatio
                # and produce thousands of unrelated pairs.
                borderline.append((i, j))
            elif ns < name_lo and os_ >= org_strong and countries_compatible(country_sets[i], country_sets[j]):
                # Acronym ↔ full-name case: name looks unrelated but very strong
                # org match (and country compatibility) suggests Claude should adjudicate.
                # Country compatibility is soft: any overlap, or one side Global/unknown,
                # counts as compatible — handles multi-country programs and missing data.
                borderline.append((i, j))
    return confirmed, borderline, all_scores


def build_judge_prompt(row_a, row_b) -> str:
    return (
        "You are deciding whether two entries refer to the same biosecurity training program.\n\n"
        f"Entry A:\n"
        f"- name: {row_a.get('name_and_title', '')}\n"
        f"- organization: {row_a.get('organisation_providing_course', '')}\n"
        f"- country: {row_a.get('country', '')}\n"
        f"- url: {row_a.get('url', '')}\n\n"
        f"Entry B:\n"
        f"- name: {row_b.get('name_and_title', '')}\n"
        f"- organization: {row_b.get('organisation_providing_course', '')}\n"
        f"- country: {row_b.get('country', '')}\n"
        f"- url: {row_b.get('url', '')}\n\n"
        "Two entries are the same program if they refer to the same offering by the same "
        "host organization. Acronyms and full names of the same program count as the same "
        "(e.g. \"ELBI\" and \"Emerging Leaders in Biosecurity Initiative\"). Different "
        "cohorts/years of the same program count as the same. Different programs at the "
        "same organization do NOT count as the same. Be conservative if uncertain."
    )


def run_judge_batch(rows, borderline_pairs, config):
    """Submit borderline pairs to Claude Batch API. Returns dict[(i,j)] -> verdict dict."""
    if not borderline_pairs:
        return {}

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    pair_by_id = {}
    batch_requests = []
    for idx, (i, j) in enumerate(borderline_pairs):
        custom_id = f"pair_{i}_{j}"[:64]
        pair_by_id[custom_id] = (i, j)
        batch_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": 512,
                "tools": [JUDGE_TOOL],
                "tool_choice": {"type": "tool", "name": "judge_duplicate"},
                "messages": [{"role": "user", "content": build_judge_prompt(rows[i], rows[j])}],
            },
        })

    print(f"Submitting Pass 2 judge batch of {len(batch_requests)} pairs...")
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Batch created: {batch.id}")

    while batch.processing_status != "ended":
        time.sleep(POLL_INTERVAL)
        batch = client.messages.batches.retrieve(batch.id)
        print(f"  Status: {batch.processing_status} | {batch.request_counts}")

    print("Pass 2 complete. Processing verdicts...")

    verdicts = {}
    for result in client.messages.batches.results(batch.id):
        custom_id = result.custom_id
        pair = pair_by_id.get(custom_id)
        if pair is None:
            continue
        if result.result.type != "succeeded":
            verdicts[pair] = {"same_program": False, "confidence": 0.0, "reasoning": "batch error"}
            continue
        try:
            tool_input = next(
                block.input for block in result.result.message.content if block.type == "tool_use"
            )
            verdicts[pair] = {
                "same_program": bool(tool_input.get("same_program", False)),
                "confidence": float(tool_input.get("confidence", 0.0)),
                "reasoning": tool_input.get("reasoning", ""),
            }
        except Exception as e:
            verdicts[pair] = {"same_program": False, "confidence": 0.0, "reasoning": f"parse error: {e}"}
    return verdicts


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[a] = b


def select_canonical(rows, indices):
    def score(idx):
        r = rows[idx]
        s = 0
        if r.get("extraction_status") == "ok":
            s += 10000
        if r.get("fetch_status") == "ok":
            s += 1000
        s += sum(1 for v in r.values() if v and str(v).strip())
        return s
    return max(indices, key=score)


def program_id_for(row) -> str:
    name = normalize_for_dedup(row.get("name_and_title", ""))
    org = normalize_for_dedup(row.get("organisation_providing_course", ""))
    sig = f"{name}|{org}"
    if not sig.strip("|"):
        return ""
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]


def merge_source_ids(rows, indices):
    parts = set()
    for i in indices:
        raw = rows[i].get("source_doc_id", "")
        if not raw:
            continue
        # Existing source_doc_id may already be pipe-joined from a prior run
        for piece in str(raw).split("|"):
            piece = piece.strip()
            if piece:
                parts.add(piece)
    return "|".join(sorted(parts))


CANDIDATE_SAMPLE_CAP = 40
CANDIDATE_CONFIRMED_QUOTA = 10  # of the cap, reserve this many for confirmed-dupe samples


def _evenly_spaced(items, n):
    """Pick n items spread evenly across the list (preserves order)."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    step = (len(items) - 1) / (n - 1) if n > 1 else 0
    return [items[round(i * step)] for i in range(n)]


def _read_existing_labels(path):
    """Read existing is_duplicate labels keyed by (url_a, url_b) so they survive regeneration."""
    if not os.path.exists(path):
        return {}
    labels = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = (r.get("is_duplicate") or "").strip()
            if not label:
                continue
            key = frozenset({r.get("url_a", ""), r.get("url_b", "")})
            labels[key] = label
    return labels


def write_candidates_csv(rows, confirmed_set, borderline_set, all_scores, verdicts, config, path):
    """Dump a small, balanced sample of algorithmic decisions for hand-labeling.

    Strategy: split the cap between (a) a sample of Pass 1 confirmed dupes — to catch
    false positives in the auto-merge zone — and (b) a sample of borderline pairs spanning
    the high end (slight rewordings) and low end (acronym-trigger cases). Sampling avoids
    flooding the labeler when there are thousands of borderline pairs.

    Existing `is_duplicate` labels are preserved across regenerations by matching on the
    url-pair key, so re-running with Pass 2 enabled doesn't wipe hand-labeled rows.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing_labels = _read_existing_labels(path)

    confirmed_sorted = sorted(confirmed_set, key=lambda p: -all_scores.get(p, (0.0, 0.0))[0])
    borderline_only = sorted(borderline_set - confirmed_set, key=lambda p: -all_scores.get(p, (0.0, 0.0))[0])

    confirmed_quota = min(CANDIDATE_CONFIRMED_QUOTA, len(confirmed_sorted), CANDIDATE_SAMPLE_CAP)
    borderline_quota = max(0, CANDIDATE_SAMPLE_CAP - confirmed_quota)

    sampled_confirmed = _evenly_spaced(confirmed_sorted, confirmed_quota)

    if borderline_quota >= len(borderline_only):
        sampled_borderline = list(borderline_only)
    else:
        head = borderline_quota // 2
        tail = borderline_quota - head
        sampled_borderline = borderline_only[:head] + borderline_only[-tail:]

    relevant = sampled_confirmed + sampled_borderline
    rows_out = []
    for pair in relevant:
        i, j = pair
        ns, os_ = all_scores.get(pair, (0.0, 0.0))
        verdict = verdicts.get(pair, {})
        in_confirmed = pair in confirmed_set
        in_pass2_dupe = (
            verdict.get("same_program") is True
            and verdict.get("confidence", 0.0) >= config["claude_confidence_threshold"]
        )
        predicted = in_confirmed or in_pass2_dupe
        url_a = rows[i].get("url", "")
        url_b = rows[j].get("url", "")
        carried_label = existing_labels.get(frozenset({url_a, url_b}), "")
        rows_out.append({
            "name_score": f"{ns:.1f}",
            "org_score": f"{os_:.1f}",
            "name_a": rows[i].get("name_and_title", ""),
            "org_a": rows[i].get("organisation_providing_course", ""),
            "country_a": rows[i].get("country", ""),
            "url_a": url_a,
            "name_b": rows[j].get("name_and_title", ""),
            "org_b": rows[j].get("organisation_providing_course", ""),
            "country_b": rows[j].get("country", ""),
            "url_b": url_b,
            "predicted_is_duplicate": "true" if predicted else "false",
            "claude_verdict": str(verdict.get("same_program")) if verdict else "",
            "claude_confidence": f"{verdict.get('confidence', 0.0):.2f}" if verdict else "",
            "is_duplicate": carried_label,
        })
    rows_out.sort(key=lambda r: (-float(r["name_score"]), -float(r["org_score"])))
    fieldnames = [
        "name_score", "org_score",
        "name_a", "org_a", "country_a", "url_a",
        "name_b", "org_b", "country_b", "url_b",
        "predicted_is_duplicate", "claude_verdict", "claude_confidence", "is_duplicate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)
    print(f"Wrote {len(rows_out)} candidate pairs to {path}")


def dedup(rows, config, run_judge=True):
    """Returns (output_rows, confirmed_set, borderline_set, all_scores, verdicts, stats)."""
    confirmed, borderline, all_scores = compute_pair_buckets(rows, config)
    print(f"Pass 1: {len(confirmed)} confirmed dupes, {len(borderline)} borderline pairs to judge")

    verdicts = {}
    if run_judge and borderline:
        verdicts = run_judge_batch(rows, borderline, config)

    confirmed_set = set(confirmed)
    borderline_set = set(borderline)
    pass2_dupes = 0
    for pair, verdict in verdicts.items():
        if verdict.get("same_program") and verdict.get("confidence", 0.0) >= config["claude_confidence_threshold"]:
            confirmed_set.add(pair)
            pass2_dupes += 1
    print(f"Pass 2: {pass2_dupes} additional dupes confirmed by Claude")

    uf = UnionFind(len(rows))
    for i, j in confirmed_set:
        uf.union(i, j)

    clusters = {}
    for i in range(len(rows)):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)

    output_rows = []
    duplicate_count_total = 0
    for indices in clusters.values():
        canonical_idx = select_canonical(rows, indices)
        canonical = dict(rows[canonical_idx])
        canonical["program_id"] = program_id_for(canonical)
        canonical["source_doc_ids"] = merge_source_ids(rows, indices)
        canonical["duplicate_count"] = len(indices) - 1
        duplicate_count_total += len(indices) - 1
        output_rows.append(canonical)

    stats = {
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "duplicates_removed": duplicate_count_total,
        "pass1_dupes": len(confirmed),
        "pass2_dupes": pass2_dupes,
        "borderline_pairs": len(borderline),
    }
    return output_rows, set(confirmed), borderline_set, all_scores, verdicts, stats


def write_output_csv(output_rows, fieldnames, stats, path):
    out_columns = list(fieldnames) + ["program_id", "source_doc_ids", "duplicate_count"]
    # Drop the original source_doc_id (replaced by source_doc_ids) only if it exists
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns, extrasaction="ignore")
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

        blank = {c: "" for c in out_columns}
        writer.writerow(blank)
        summary = [
            ("Input rows (Stage 3)", stats["input_rows"]),
            ("Unique programs", stats["output_rows"]),
            ("Duplicates removed", stats["duplicates_removed"]),
            ("Pass 1 (heuristic) dupes", stats["pass1_dupes"]),
            ("Pass 2 (Claude judge) dupes", stats["pass2_dupes"]),
        ]
        for label, value in summary:
            row = {c: "" for c in out_columns}
            row["url"] = label
            row["source_doc_id"] = str(value)
            writer.writerow(row)
    print(f"Wrote {len(output_rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-candidates",
        action="store_true",
        help="Also write candidate pairs CSV for labeling",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip Pass 2 (Claude judge); useful for quick iteration",
    )
    args = parser.parse_args()

    if not os.path.exists(INPUT_CSV):
        print(f"No input CSV at {INPUT_CSV}. Run stage3_extract.py first.")
        sys.exit(1)

    config = load_config()
    print(f"Config: {config}")

    rows, fieldnames = read_stage3_csv(INPUT_CSV)
    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    output_rows, confirmed_set, borderline_set, all_scores, verdicts, stats = dedup(
        rows, config, run_judge=not args.no_judge
    )

    os.makedirs("output", exist_ok=True)
    write_output_csv(output_rows, fieldnames, stats, OUTPUT_CSV)

    if args.dump_candidates:
        write_candidates_csv(rows, confirmed_set, borderline_set, all_scores, verdicts, config, CANDIDATES_CSV)

    print("\n--- Stage 4 Summary ---")
    print(f"  Input rows:       {stats['input_rows']}")
    print(f"  Unique programs:  {stats['output_rows']}")
    print(f"  Removed dupes:    {stats['duplicates_removed']}")
    print(f"  Pass 1 dupes:     {stats['pass1_dupes']}")
    print(f"  Pass 2 dupes:     {stats['pass2_dupes']} (of {stats['borderline_pairs']} borderline)")


if __name__ == "__main__":
    main()
