"""The quality gate (D-032): the model is published only if it passes every line.

The thresholds were written in the decision log and in `src/final_model.py` BEFORE the
test set was opened, like a pass mark set before the exams are marked (course, W9).
This script reads `models/metrics.json` (written by `train.py`), writes the release
notes `models/report.md` (also shown in the GitHub job summary) and exits with 1 if a
line fails, which stops the pipeline before anything is published.

    python check_quality.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from final_model import QUALITY_GATE  # noqa: E402

MODELS = ROOT / "models"


def main():
    """Check the gate, write the report, return the exit code."""
    m = json.loads((MODELS / "metrics.json").read_text())
    v = m["gate"]["values"]
    lines = [
        ("PR-AUC on test", v["pr_auc"], f">= {QUALITY_GATE['min_pr_auc']:.2f}",
         v["pr_auc"] >= QUALITY_GATE["min_pr_auc"], "{:.3f}"),
        ("PR-AUC gain over the LACE index", v["gain_over_lace"],
         f">= {QUALITY_GATE['min_gain_over_lace']:.2f}",
         v["gain_over_lace"] >= QUALITY_GATE["min_gain_over_lace"], "{:+.3f}"),
        ("Share of readmissions in the High tier", v["recall_top10"],
         f">= {QUALITY_GATE['min_recall_top10']:.2f}",
         v["recall_top10"] >= QUALITY_GATE["min_recall_top10"], "{:.3f}"),
        ("Calibration gap, mean risk against observed rate", v["calibration_gap_pp"],
         f"<= {QUALITY_GATE['max_calibration_gap_pp']:.1f} pp",
         v["calibration_gap_pp"] <= QUALITY_GATE["max_calibration_gap_pp"],
         "{:.2f} pp"),
    ]
    passed = all(line[3] for line in lines)
    t, r = m["test"], m["reached_at_10pct"]
    report = [
        f"# Diabetes readmission model {m['version']}",
        "",
        f"**Quality gate: {'PASSED' if passed else 'FAILED'}** "
        f"(thresholds fixed before the test set was opened, decision log D-032)",
        "",
        "| Check | Value | Threshold | Result |",
        "|---|---:|---:|---|",
    ]
    for name, value, threshold, ok, fmt in lines:
        report.append(f"| {name} | {fmt.format(value)} | {threshold} | "
                      f"{'pass' if ok else 'FAIL'} |")
    report += [
        "",
        f"Test patients: {t['encounters']:,} discharges of {t['patients']:,} patients, "
        f"{t['readmissions']:,} readmitted in < 30 days ({100 * t['rate']:.1f}%).",
        "",
        f"With capacity to follow up 10% of the discharges ({m['calls_at_10pct']:,} "
        f"calls): {r['XGBoost (final)']:,} readmissions reached, against "
        f"{r['LACE index']:,} with the LACE index and {r['Random order']:.0f} at random.",
        "",
        f"Model: {m['model']}, fitted on {m['fitted_on']['encounters']:,} discharges; "
        f"created {m['created']}.",
    ]
    text = "\n".join(report) + "\n"
    (MODELS / "report.md").write_text(text, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
