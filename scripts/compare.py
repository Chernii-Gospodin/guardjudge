"""Build a Markdown comparison table from several evaluate.py reports.

  python scripts/compare.py results/qwen3_before.json results/qwen3_after.json \
      results/qwen3guard_before.json results/qwen3guard_after.json --out results/summary.md

Order of the files = order of the rows. Adds per-source and per-language F1 tables.
"""
import argparse
import json
from pathlib import Path


def f(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def build(reports: list[dict]) -> str:
    lines = ["## Overall", "",
             "| model | n | precision | recall | F1 | FPR | parse fail | p50, s | p95, s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        m = r["overall"]
        lines.append(f"| {r['name']} | {m['n']} | {f(m['precision'])} | {f(m['recall'])} | **{f(m['f1'])}** "
                     f"| {f(m['fpr'])} | {m['parse_fail']} | {f(m['latency_p50_s'])} | {f(m['latency_p95_s'])} |")

    for key, title in (("by_source", "F1 by source"), ("by_language", "F1 by language"),
                       ("recall_by_category", "Recall by attack category")):
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
