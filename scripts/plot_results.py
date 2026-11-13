"""Per-category plots from evaluate.py reports.

  python scripts/plot_results.py results/*.json --out results/plots

Writes:
  category_detection.png  share of attacks caught per attack family (binary verdict), all models
  category_prf.png        precision / recall / F1 per class (none + attack categories), judge-format models
  confusion_<n>.png       row-normalised confusion matrix (gold -> predicted class) per judge-format model

Category-level scores are recomputed from saved predictions (evaluate.py --save-predictions).
Native guard formats (Qwen3Guard, Llama Guard) use their own taxonomy: they appear only in the
detection plot.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from guardjudge.metrics import UNSPECIFIED, category_metrics, recall_by_category  # noqa: E402
from guardjudge.schema import CATEGORIES  # noqa: E402

KNOWN = set(CATEGORIES) - {"none"}


def class_order(classes):
    return sorted(classes, key=lambda c: (c != "none", c == UNSPECIFIED, c))


def bars(ax, groups, series, ylabel, title):
    """Grouped bars: one group per x label, one bar per series (model)."""
    x = np.arange(len(groups))
    w = 0.8 / max(len(series), 1)
    for i, (name, vals) in enumerate(series):
        b = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name)
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def plot_detection(reports, out: Path) -> Path | None:
    per = [(r["name"], recall_by_category(r["predictions"]) if r.get("predictions")
            else r.get("recall_by_category", {})) for r in reports]
    cats = sorted({c for _, d in per for c in d})
    if not cats:
        return None
    n_by_cat = {}
    for r in reports:
        for p in r.get("predictions", []):
            if p["gold"]:
                n_by_cat[p["category"]] = n_by_cat.get(p["category"], 0) + 1
        if n_by_cat:
            break
    labels = [f"{c}\n(n={n_by_cat[c]})" if c in n_by_cat else c for c in cats]
    fig, ax = plt.subplots(figsize=(max(7, 2.6 * len(cats) + 2), 4.5))
    bars(ax, labels, [(name, [d.get(c, 0) for c in cats]) for name, d in per],
         "доля пойманных атак (recall)", "Детекция по семействам атак (бинарный вердикт)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    path = out / "category_detection.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def judge_metrics(reports):
    out = []
    for r in reports:
        if r.get("prompt_style", "judge") != "judge":
            continue
        cm = category_metrics(r["predictions"], known=KNOWN) if r.get("predictions") else r.get("category_metrics")
        if cm:
            out.append((r["name"], cm))
    return out


def plot_prf(cms, out: Path) -> Path | None:
    if not cms:
        return None
    classes = class_order({c for _, cm in cms for c in cm["per_class"]})
    fig, axes = plt.subplots(1, 3, figsize=(max(12, 4.2 * len(classes)), 4.6), sharey=True)
    for ax, metric, title in zip(axes, ("precision", "recall", "f1"), ("Precision", "Recall", "F1")):
        series = [(name, [cm["per_class"].get(c, {}).get(metric, 0) for c in classes]) for name, cm in cms]
        support = {c: next(cm["per_class"][c]["support"] for _, cm in cms if c in cm["per_class"]) for c in classes}
        bars(ax, [f"{c}\n(n={support[c]})" for c in classes], series, "" if metric != "precision" else "значение",
             f"{title} по классам")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=min(len(cms), 3))
    fig.suptitle("Качество по категориям (класс = none или категория атаки, которую назвал судья)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path = out / "category_prf.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusions(cms, out: Path) -> list[Path]:
    paths = []
    for i, (name, cm) in enumerate(cms, 1):
        gold = class_order(cm["classes"])
        pred = class_order(set(gold) | {p for row in cm["confusion"].values() for p in row})
        counts = np.array([[cm["confusion"].get(g, {}).get(p, 0) for p in pred] for g in gold], dtype=float)
        norm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(1.6 * len(pred) + 2.5, 1.0 * len(gold) + 2))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (y, x), v in np.ndenumerate(norm):
            ax.text(x, y, f"{v:.2f}\n({int(counts[y, x])})", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "black")
        ax.set_xticks(range(len(pred)), pred, rotation=20, ha="right")
        ax.set_yticks(range(len(gold)), gold)
        ax.set_xlabel("предсказано")
        ax.set_ylabel("правильный класс")
        ax.set_title(f"{name}\nmacro-F1={cm['macro_f1']:.3f}, верная категория среди пойманных="
                     f"{cm['category_accuracy_on_detected'] if cm['category_accuracy_on_detected'] is not None else '-'}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower() or "model"
        path = out / f"confusion_{i}_{slug}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", default="results/plots")
    a = ap.parse_args()
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.reports if Path(p).is_file()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cms = judge_metrics(reports)
    written = [plot_detection(reports, out), plot_prf(cms, out), *plot_confusions(cms, out)]
    for p in written:
        if p:
            print(p)


if __name__ == "__main__":
    main()
