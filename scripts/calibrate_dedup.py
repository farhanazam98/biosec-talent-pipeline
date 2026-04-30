"""
Calibration harness for Stage 4 deduplication.

Reads `tests/fixtures/duplicate_candidates.csv` (after the user has filled in
the `is_duplicate` column) and sweeps Pass 1 (name/org threshold) and Pass 2
(Claude confidence threshold) settings to find the combo with best F1.

Usage:
    python scripts/calibrate_dedup.py
"""

import csv
import os
import sys

import yaml

CANDIDATES_CSV = "tests/fixtures/duplicate_candidates.csv"
REPORT_PATH = "docs/dedup_calibration.md"
CONFIG_PATH = "config/dedup.yaml"

NAME_THRESHOLDS = [85, 88, 90, 92, 95]
ORG_THRESHOLDS = [70, 80, 90]
CLAUDE_THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def parse_bool(s):
    s = (s or "").strip().lower()
    if s in {"true", "yes", "y", "1", "t"}:
        return True
    if s in {"false", "no", "n", "0", "f"}:
        return False
    return None


def parse_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_labeled_pairs(path):
    if not os.path.exists(path):
        print(f"No candidate CSV at {path}. Run stage4_dedup.py --dump-candidates first.", file=sys.stderr)
        sys.exit(1)
    pairs = []
    skipped_na = 0
    skipped_blank = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("is_duplicate") or "").strip().lower()
            label = parse_bool(raw)
            if label is None:
                if raw in {"n/a", "na", "?"}:
                    skipped_na += 1
                else:
                    skipped_blank += 1
                continue
            pairs.append({
                "name_score": parse_float(row.get("name_score")) or 0.0,
                "org_score": parse_float(row.get("org_score")) or 0.0,
                "claude_verdict": parse_bool(row.get("claude_verdict")),
                "claude_confidence": parse_float(row.get("claude_confidence")) or 0.0,
                "expected_is_duplicate": label,
                "name_a": row.get("name_a", ""),
                "name_b": row.get("name_b", ""),
            })
    if skipped_na or skipped_blank:
        print(f"Excluded from calibration: {skipped_na} ambiguous (n/a), {skipped_blank} unlabeled")
    return pairs


def predict(pair, name_thresh, org_thresh, claude_thresh):
    if pair["name_score"] >= name_thresh and pair["org_score"] >= org_thresh:
        return True
    if pair["claude_verdict"] is True and pair["claude_confidence"] >= claude_thresh:
        return True
    return False


def evaluate(pairs, name_thresh, org_thresh, claude_thresh):
    tp = fp = fn = tn = 0
    for p in pairs:
        pred = predict(p, name_thresh, org_thresh, claude_thresh)
        actual = p["expected_is_duplicate"]
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def main():
    pairs = load_labeled_pairs(CANDIDATES_CSV)
    if not pairs:
        print(f"No labeled rows in {CANDIDATES_CSV}. Fill in the `is_duplicate` column first.", file=sys.stderr)
        sys.exit(1)
    n_pos = sum(1 for p in pairs if p["expected_is_duplicate"])
    n_neg = len(pairs) - n_pos
    print(f"Loaded {len(pairs)} labeled pairs ({n_pos} dupes, {n_neg} non-dupes)")

    grid = []
    for nt in NAME_THRESHOLDS:
        for ot in ORG_THRESHOLDS:
            for ct in CLAUDE_THRESHOLDS:
                metrics = evaluate(pairs, nt, ot, ct)
                grid.append((nt, ot, ct, metrics))

    # Round F1 to 4 decimals so floating-point hairs don't override the FP tiebreaker.
    # Then prefer fewer FPs (merging distinct programs is worse than missing a dupe),
    # then more conservative thresholds (higher name, then higher claude_confidence).
    grid.sort(key=lambda x: (
        -round(x[3]["f1"], 4),
        x[3]["fp"],
        -x[0],   # prefer higher name threshold
        -x[2],   # prefer higher claude threshold
    ))
    best = grid[0]

    print("\n--- Top 10 Threshold Combos by F1 ---")
    print(f"{'Name':<6} {'Org':<6} {'Claude':<8} {'TP':<4} {'FP':<4} {'FN':<4} {'TN':<4} {'Precision':<10} {'Recall':<8} {'F1'}")
    print("-" * 80)
    for nt, ot, ct, m in grid[:10]:
        print(
            f"{nt:<6} {ot:<6} {ct:<8.2f} "
            f"{m['tp']:<4} {m['fp']:<4} {m['fn']:<4} {m['tn']:<4} "
            f"{m['precision']:<10.3f} {m['recall']:<8.3f} {m['f1']:.3f}"
        )

    print("\n--- Recommendation ---")
    nt, ot, ct, m = best
    print(f"name_threshold={nt}, org_threshold={ot}, claude_confidence={ct:.2f}")
    print(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")
    print(f"  tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Dedup Calibration Report\n\n")
        f.write(f"Labeled pairs: {len(pairs)} ({n_pos} dupes, {n_neg} non-dupes)\n\n")
        f.write("## Top 20 Threshold Combos by F1\n\n")
        f.write("| Name | Org | Claude | TP | FP | FN | TN | Precision | Recall | F1 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for nt_, ot_, ct_, m_ in grid[:20]:
            f.write(
                f"| {nt_} | {ot_} | {ct_:.2f} | "
                f"{m_['tp']} | {m_['fp']} | {m_['fn']} | {m_['tn']} | "
                f"{m_['precision']:.3f} | {m_['recall']:.3f} | {m_['f1']:.3f} |\n"
            )
        f.write("\n## Recommendation\n\n")
        f.write("Paste into `config/dedup.yaml`:\n\n")
        f.write("```yaml\n")
        f.write(f"heuristic_name_threshold: {nt}\n")
        f.write(f"heuristic_org_threshold: {ot}\n")
        f.write(f"claude_confidence_threshold: {ct:.2f}\n")
        f.write("```\n\n")
        f.write(f"Performance: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}\n")
        f.write(f"(tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, tn={m['tn']})\n")
    print(f"\nReport written to {REPORT_PATH}")

    # Merge recommended values into existing config so non-calibrated keys
    # (borderline_name_lower, borderline_org_strong) are preserved.
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
    config["heuristic_name_threshold"] = nt
    config["heuristic_org_threshold"] = ot
    config["claude_confidence_threshold"] = round(ct, 2)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote thresholds to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
