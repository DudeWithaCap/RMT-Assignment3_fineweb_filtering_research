"""Static reproducible PNG/SVG research figures, using only measured data."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from src.filter_assets import ROOT, sha256_file
from .preserve import OUTPUT, verify_preserved

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze import read_inputs, CONDITIONS, FILTERS

LABELS = {"baseline": "Baseline", "url": "URL", "language": "Language", "quality": "Quality", "dedup": "Dedup"}
COLORS = {"baseline": "#222831", "url": "#277da8", "language": "#159474", "quality": "#dc8e26", "dedup": "#9356a0"}
DATASET_COLORS = ["#285f8d", "#259787", "#e3a23b"]
DATASETS = (1, 2, 3)


def ecdf(histogram):
    x = np.array(sorted(histogram), dtype=float)
    weights = np.array([histogram[int(value)] for value in x], dtype=float)
    return x, np.cumsum(weights) / weights.sum()


def figures(directory=OUTPUT / "figures", analysis_directory=OUTPUT):
    directory, analysis_directory = Path(directory), Path(analysis_directory)
    if not directory.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Figures must stay under results/stage5")
    verify_preserved()
    directory.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((analysis_directory / "analysis.json").read_text(encoding="utf-8"))
    metrics, histograms, _, _ = read_inputs()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "svg.hashsalt": "fineweb-stage5-fixed", "figure.dpi": 110})
    manifest = []

    def save(figure, name, caption):
        for extension in ("png", "svg"):
            path = directory / f"{name}.{extension}"
            metadata = {"Software": "FineWeb research Stage 5"} if extension == "png" else {"Date": None, "Creator": "FineWeb research Stage 5"}
            figure.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata)
        plt.close(figure)
        manifest.append({"name": name, "caption": caption,
                         "png_sha256": sha256_file(directory / f"{name}.png"), "svg_sha256": sha256_file(directory / f"{name}.svg")})

    for field, name, title in (("document_reduction_percentage", "document_reduction", "Document reduction by independent filter"),
                                ("token_reduction_percentage", "token_reduction", "Token reduction by independent filter")):
        figure, axis = plt.subplots(figsize=(10.5, 5), layout="constrained")
        positions = np.arange(len(FILTERS))
        for index, dataset in enumerate(DATASETS):
            values = [metrics[(dataset, condition)][field] for condition in FILTERS]
            bars = axis.bar(positions + (index - 1) * .24, values, .23, label=f"Dataset {dataset}", color=DATASET_COLORS[index])
            axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], fontsize=9, padding=3)
        axis.set_xticks(positions, [LABELS[c] for c in FILTERS])
        axis.set_ylim(0, 108)
        axis.set_ylabel("Reduction from matching baseline (%)")
        axis.set_title(title, loc="left", pad=15)
        axis.grid(axis="y", alpha=.18)
        axis.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.01))
        save(figure, name, "Each bar is a separate observed dataset; no population error bars or sequential filtering.")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharey=True, layout="constrained")
    for axis, dataset in zip(axes, DATASETS):
        for condition in CONDITIONS:
            x, y = ecdf(histograms[(dataset, condition)])
            axis.step(x, y, where="post", color=COLORS[condition], linewidth=1.9 if condition == "baseline" else 1.4,
                      linestyle="--" if condition == "dedup" else ":" if condition == "url" else "-", label=LABELS[condition])
        axis.set_xscale("log")
        axis.set_ylim(0, 1.025)
        axis.set_title(f"Dataset {dataset}")
        axis.set_xlabel("Sentence length (tokens; log scale)")
        axis.grid(alpha=.15)
    axes[0].set_ylabel("Fraction of observed sentences ≤ length")
    axes[2].legend(frameon=False, loc="lower right")
    figure.suptitle("Full empirical sentence-length distributions — primary saved text", fontsize=14)
    save(figure, "sentence_ecdf", "All lengths are included. Curves are descriptive; pooled sentences are not independent test observations.")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True, layout="constrained")
    for axis, dataset in zip(axes, DATASETS):
        for index, condition in enumerate(CONDITIONS):
            row = metrics[(dataset, condition)]
            axis.hlines(index, row["sentence_p25_tokens"], row["sentence_p75_tokens"], color=COLORS[condition], linewidth=6, alpha=.6)
            axis.plot(row["sentence_median_tokens"], index, "o", color=COLORS[condition], markersize=7)
            axis.plot(row["sentence_mean_tokens"], index, "x", color=COLORS[condition], markersize=8, markeredgewidth=2)
        axis.set_yticks(np.arange(5), [LABELS[c] for c in CONDITIONS])
        axis.set_xlim(0, 49)
        axis.set_xlabel("Sentence length (tokens)")
        axis.set_title(f"Dataset {dataset}")
        axis.grid(axis="x", alpha=.17)
    axes[0].invert_yaxis()
    figure.suptitle("Sentence summaries: bars = 25th–75th percentiles; dots = median; × = mean", fontsize=13)
    save(figure, "sentence_location_spread", "Central quantiles and means only; the full tails appear in the ECDF figure. No confidence intervals are shown.")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    positions = np.arange(5)
    for axis, field, title, ylabel in ((axes[0], "unique_domains", "Hostname richness", "Unique hostnames"),
                                       (axes[1], "domain_entropy_bits", "Shannon entropy", "Entropy (bits)")):
        for index, dataset in enumerate(DATASETS):
            axis.bar(positions + (index - 1) * .24, [metrics[(dataset, c)][field] for c in CONDITIONS], .23,
                     color=DATASET_COLORS[index], label=f"Dataset {dataset}")
        axis.set_xticks(positions, [LABELS[c] for c in CONDITIONS], rotation=20)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.17)
    axes[1].legend(frameon=False)
    figure.suptitle("Domain diversity under independent conditions", fontsize=14)
    save(figure, "domain_diversity", "Raw hostname counts and document-weighted entropy; lower retained sample sizes constrain attainable richness and entropy.")

    effect_index = {(row["dataset"], row["filter"]): row for row in analysis["sentence_effects"]}
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.2), layout="constrained")
    for axis, field, title, maximum, decimals in (
            (axes[0, 0], "document_reduction_percentage", "Document reduction (%)", 100, 1),
            (axes[0, 1], "token_reduction_percentage", "Token reduction (%)", 100, 1),
            (axes[1, 0], "domain_retention_percentage", "Hostname retention (%)", 100, 1),
            (axes[1, 1], "cdf_sup_distance", "Sentence CDF distance (descriptive)", None, 4)):
        values = np.array([[effect_index[(d, c)][field] for c in FILTERS] for d in DATASETS])
        plot = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=maximum, aspect="auto")
        axis.set_xticks(np.arange(4), [LABELS[c] for c in FILTERS])
        axis.set_yticks(np.arange(3), [f"Dataset {d}" for d in DATASETS])
        axis.set_title(title, pad=12)
        for row in range(3):
            for col in range(4):
                threshold = (maximum if maximum else values.max()) * .55
                axis.text(col, row, f"{values[row,col]:.{decimals}f}", ha="center", va="center", color="white" if values[row,col] > threshold else "#17202a")
        figure.colorbar(plot, ax=axis, shrink=.8)
    figure.suptitle("Cross-dataset effect sizes — each row retains its own baseline", fontsize=14)
    save(figure, "cross_dataset_comparison", "Color scales differ by metric; values are observed effects, not significance scores.")

    size_index = {(row["dataset"], row["filter"]): row for row in analysis["domain_size_adjustment"]}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for index, dataset in enumerate(DATASETS):
        x = np.arange(4) + (index - 1) * .06
        axes[0].plot(x, [size_index[(dataset, c)]["observed_over_expected_domains"] for c in FILTERS], "o-", color=DATASET_COLORS[index], label=f"Dataset {dataset}")
        axes[1].plot(x, [size_index[(dataset, c)]["observed_minus_expected_entropy_bits"] for c in FILTERS], "o-", color=DATASET_COLORS[index])
    axes[0].axhline(1, color="#4c5663", linestyle="--", linewidth=1)
    axes[1].axhline(0, color="#4c5663", linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xticks(np.arange(4), [LABELS[c] for c in FILTERS])
        axis.grid(alpha=.16)
    axes[0].set_ylabel("Observed / expected unique hostnames")
    axes[1].set_ylabel("Observed − expected entropy (bits)")
    axes[0].set_title("Richness relative to same-size subsets")
    axes[1].set_title("Entropy relative to same-size subsets")
    axes[0].legend(frameon=False)
    figure.suptitle("Exact uniform-subset benchmark within each fixed baseline", fontsize=14)
    save(figure, "domain_size_adjusted", "Benchmarks condition on the observed baseline and retained count. They are not population expectations, p-values, or confidence intervals.")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for index, summary in enumerate(analysis["c4_summary"]):
        axes[0].bar(index - .17, summary["saved_tokens"], .32, color="#dc8e26", label="Saved quality" if index == 0 else None)
        axes[0].bar(index + .17, summary["cleaned_tokens"], .32, color="#315b75", label="After C4" if index == 0 else None)
        changes = Counter(-row["token_change"] for row in analysis["c4_paired_documents"] if row["dataset"] == summary["dataset"])
        x, y = ecdf(changes)
        axes[1].step(x, y, where="post", color=DATASET_COLORS[index], label=f"Dataset {summary['dataset']}")
    axes[0].set_xticks(np.arange(3), [f"Dataset {d}" for d in DATASETS])
    axes[0].set_ylabel("Total measured tokens")
    axes[0].set_title("Same retained document IDs")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Tokens removed by C4 per retained document")
    axes[1].set_ylabel("Fraction of paired documents ≤ loss")
    axes[1].set_title("Document-paired transformation effects")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=.16)
    figure.suptitle("Supplemental C4 cleanup — separate from primary quality selection", fontsize=14)
    save(figure, "c4_transformation", "Original quality outputs stay unchanged. Per-document token changes use paired IDs rather than independent samples.")
    (directory / "figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify_preserved()
    print(f"Created {len(manifest)} measured figures in PNG and SVG")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT / "figures")
    parser.add_argument("--analysis-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    figures(args.output_dir, args.analysis_dir)
