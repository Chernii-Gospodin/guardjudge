"""Build a Markdown comparison table from several evaluate.py reports.

  python scripts/compare.py results/qwen3_before.json results/qwen3_after.json \
      results/qwen3guard_before.json results/qwen3guard_after.json --out results/summary.md

Order of the files = order of the rows. Tables: binary decision (safe/unsafe), F1 by source and
language, binary recall per attack family, and category quality (did the judge also name the right
category) with a confusion matrix per model.

Category metrics are recomputed from the saved per-example predictions (--save-predictions), so
reports produced before they were added still get them.
"""
import argparse
import json
from pathlib import Path

from guardjudge.metrics import category_metrics
from guardjudge.schema import CATEGORIES


def f(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def get_category_metrics(r: dict) -> dict | None:
    """Category metrics of a report; None for native guard formats (own taxonomy)."""
    if r.get("prompt_style", "judge") != "judge":
        return None
    if r.get("predictions"):
        return category_metrics(r["predictions"], known=set(CATEGORIES) - {"none"})
    return r.get("category_metrics")


def category_section(reports: list[dict]) -> list[str]:
    cms = [(r, get_category_metrics(r)) for r in reports]
    classes = sorted({c for _, cm in cms if cm for c in cm["classes"]}, key=lambda c: (c != "none", c))
    if not classes:
        return []
    lines = ["", "## Category (does the judge name the right attack family?)", "",
             "*correct cat.* — among attacks the judge flagged, share with the right category; "
             "*accuracy / macro-F1* — classes `none` + attack categories. "
             "Native guard formats use their own taxonomy and are not scored (—).", "",
             "| model | detected | correct cat. | accuracy | macro-F1 | "
             + " | ".join(f"F1 {c}" for c in classes) + " |",
             "|---|---|---|---|---|" + "---|" * len(classes)]
    for r, cm in cms:
        if not cm:
            lines.append(f"| {r['name']} | " + " | ".join(["—"] * (4 + len(classes))) + " |")
            continue
        per = [f(cm["per_class"][c]["f1"]) if c in cm["per_class"] else "-" for c in classes]
        lines.append(f"| {r['name']} | {cm['detected_n']} | {f(cm['category_accuracy_on_detected'])} | "
                     f"{f(cm['accuracy'])} | **{f(cm['macro_f1'])}** | " + " | ".join(per) + " |")
    for r, cm in cms:
        if not cm:
            continue
        preds = classes + sorted({p for row in cm["confusion"].values() for p in row} - set(classes))
        lines += ["", f"**Confusion: {r['name']}** (rows = gold, columns = predicted)", "",
                  "| gold / pred | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
        for g in cm["classes"]:
            lines.append(f"| {g} | " + " | ".join(str(cm["confusion"].get(g, {}).get(p, 0)) for p in preds) + " |")
    return lines


def build(reports: list[dict]) -> str:
    lines = ["## Overall", "",
             "| model | n | precision | recall | F1 | FPR | parse fail | p50, s | p95, s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        m = r["overall"]
        lines.append(f"| {r['name']} | {m['n']} | {f(m['precision'])} | {f(m['recall'])} | **{f(m['f1'])}** "
                     f"| {f(m['fpr'])} | {m['parse_fail']} | {f(m['latency_p50_s'])} | {f(m['latency_p95_s'])} |")

    for key, title in (("by_source", "F1 by source"), ("by_language", "F1 by language"),
                       ("recall_by_category", "Detection rate (binary recall) by attack family")):
        groups = sorted({g for r in reports for g in r.get(key, {})})
        if not groups:
            continue
        lines += ["", f"## {title}", "", "| model | " + " | ".join(groups) + " |",
                  "|---|" + "---|" * len(groups)]
        for r in reports:
            vals = []
            for g in groups:
                v = r.get(key, {}).get(g)
                vals.append(f(v) if key == "recall_by_category" else (f(v["f1"]) if v else "-"))
            lines.append(f"| {r['name']} | " + " | ".join(vals) + " |")
        if key != "recall_by_category":
            sizes = [str(next((r[key][g]["n"] for r in reports if g in r.get(key, {})), "-")) for g in groups]
            lines.append("| *n* | " + " | ".join(sizes) + " |")
    lines += category_section(reports)
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reports = []
    for p in a.reports:
        if not Path(p).is_file():
            print(f"[skip] {p} not found")
            continue
        reports.append(json.loads(Path(p).read_text(encoding="utf-8")))
    md = build(reports)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
