"""Cross-filter overlap of removed-document sets in completed independent runs.

Reads the per-filter decision logs of an independent run (every filter judged the
same immutable baseline), verifies them, and measures how far the filters remove
the same documents. Results go to a new dated folder; source runs are never
modified. No filter is executed here.
"""

import argparse
from collections import Counter
import csv
from itertools import combinations
import json
import math
import os
from pathlib import Path

from .filter_assets import ROOT, sha256_file
from .metrics.calculations import entropy_bits, histogram_quantile, sentence_statistics, safe_percentage
from .run_filters import STAGES, baseline_rows
from .run_results import LOCAL_TIME, create_run, now, write_json

LABELS = {"url": "URL", "language": "Language", "quality": "Quality", "dedup": "Dedup"}
# Validated categorical slots (dataviz reference palette 1-4), fixed per filter.
COLORS = {"url": "#2a78d6", "language": "#eb6834", "quality": "#1baf7a", "dedup": "#eda100", "baseline": "#52514e"}
PATTERNS = range(1 << len(STAGES))
PAIRS = tuple(combinations(STAGES, 2))
METRICS = ("jaccard", "containment_a_in_b", "containment_b_in_a", "lift", "phi")


# ---------------------------------------------------------------------------
# Loading and verification


def load_run(directory):
    """Return verified per-document removal patterns for one independent run."""
    directory = Path(directory).resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("filter_mode") != "independent":
        raise ValueError(f"{directory.name}: overlap needs a complete independent run")
    if sorted(summary.get("filter_order", ())) != sorted(STAGES):
        raise ValueError(f"{directory.name}: all four filters must have been run independently")
    baseline_hash = summary["baseline_sha256"]
    ids, patterns, sources = None, None, {}
    for bit, stage in enumerate(STAGES):
        report_path = directory / "details/results" / f"{stage}_summary.json"
        decisions_path = directory / "details/results" / f"{stage}_decisions.jsonl"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["status"] != "complete" or report["baseline_sha256"] != baseline_hash
                or report["baseline_sha256_after"] != baseline_hash):
            raise ValueError(f"{directory.name}/{stage}: filter did not judge the run's baseline")
        if sha256_file(decisions_path) != report["decisions_sha256"]:
            raise ValueError(f"{directory.name}/{stage}: decision log does not match its recorded hash")
        stage_ids, removed_rows = [], []
        with decisions_path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                decision = json.loads(line)
                if decision["stage"] != stage or type(decision["kept"]) is not bool:
                    raise ValueError(f"{directory.name}/{stage}: malformed decision at row {index + 1}")
                stage_ids.append(decision["document_id"])
                if not decision["kept"]:
                    removed_rows.append(index)
        if ids is None:
            ids, patterns = stage_ids, [0] * len(stage_ids)
        elif stage_ids != ids:
            raise ValueError(f"{directory.name}/{stage}: decision IDs differ from other filters' order")
        for index in removed_rows:
            patterns[index] |= 1 << bit
        removed = len(removed_rows)
        if (len(stage_ids) != report["input_count"] or removed != report["removed_count"]
                or len(set(stage_ids)) != len(stage_ids)):
            raise ValueError(f"{directory.name}/{stage}: decision counts do not reconcile")
        sources[stage] = {"summary": str(report_path), "decisions_sha256": report["decisions_sha256"],
                          "removed": removed}
    if len(ids) != summary["baseline_documents"]:
        raise ValueError(f"{directory.name}: decision logs do not cover the whole baseline")
    config = json.loads((directory / "details/filter_config.json").read_text(encoding="utf-8"))
    return {"run": directory, "dataset": summary["dataset"], "baseline": Path(summary["baseline"]),
            "baseline_sha256": baseline_hash, "assets_directory": config.get("assets_directory"),
            "ids": ids, "patterns": patterns, "sources": sources}


# ---------------------------------------------------------------------------
# Overlap statistics (all derived from 16 pattern counts)


def removed_by(pattern_counts, stage):
    bit = 1 << STAGES.index(stage)
    return sum(count for pattern, count in pattern_counts.items() if pattern & bit)


def pair_statistics(pattern_counts, a, b):
    bit_a, bit_b = 1 << STAGES.index(a), 1 << STAGES.index(b)
    n11 = n10 = n01 = n00 = 0
    for pattern, count in pattern_counts.items():
        in_a, in_b = bool(pattern & bit_a), bool(pattern & bit_b)
        if in_a and in_b:
            n11 += count
        elif in_a:
            n10 += count
        elif in_b:
            n01 += count
        else:
            n00 += count
    total = n11 + n10 + n01 + n00
    size_a, size_b, union = n11 + n10, n11 + n01, n11 + n10 + n01
    expected = size_a * size_b / total if total else 0
    denominator = math.sqrt(size_a * size_b * (total - size_a) * (total - size_b))
    return {"filter_a": a, "filter_b": b, "removed_a": size_a, "removed_b": size_b,
            "both_removed": n11, "only_a": n10, "only_b": n01, "neither_removed": n00, "union_removed": union,
            "jaccard": n11 / union if union else None,
            "containment_a_in_b": n11 / size_a if size_a else None,
            "containment_b_in_a": n11 / size_b if size_b else None,
            "expected_both_if_independent": expected,
            "lift": n11 / expected if expected else None,
            "phi": (n11 * n00 - n10 * n01) / denominator if denominator else None}


def bootstrap(pattern_counts, iterations, seed):
    """Document-level percentile bootstrap: resampling documents is multinomial over patterns."""
    import numpy as np

    total = sum(pattern_counts.values())
    if iterations <= 0 or total == 0:
        return {}
    probabilities = np.array([pattern_counts.get(p, 0) for p in PATTERNS], dtype=float) / total
    samples = np.random.default_rng(seed).multinomial(total, probabilities, size=iterations)
    intervals = {}
    for a, b in PAIRS:
        bit_a, bit_b = 1 << STAGES.index(a), 1 << STAGES.index(b)
        mask = lambda wa, wb: np.array([bool(p & bit_a) == wa and bool(p & bit_b) == wb for p in PATTERNS])
        n11, n10 = samples[:, mask(True, True)].sum(1), samples[:, mask(True, False)].sum(1)
        n01, n00 = samples[:, mask(False, True)].sum(1), samples[:, mask(False, False)].sum(1)
        size_a, size_b = n11 + n10, n11 + n01
        with np.errstate(divide="ignore", invalid="ignore"):
            values = {"jaccard": n11 / (n11 + n10 + n01),
                      "containment_a_in_b": n11 / size_a, "containment_b_in_a": n11 / size_b,
                      "lift": n11 * total / (size_a * size_b),
                      "phi": (n11 * n00 - n10 * n01) / np.sqrt(
                          size_a.astype(float) * size_b * (total - size_a) * (total - size_b))}
        for metric, array in values.items():
            finite = array[np.isfinite(array)]
            intervals[(a, b, metric)] = ((float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))
                                         if len(finite) == iterations else (None, None))
    return intervals


def pattern_rows(pattern_counts, total):
    rows = []
    for pattern in PATTERNS:
        filters = [stage for bit, stage in enumerate(STAGES) if pattern & (1 << bit)]
        count = pattern_counts.get(pattern, 0)
        rows.append({"pattern": "+".join(filters) or "none", "removed_by_count": len(filters),
                     **{stage: int(stage in filters) for stage in STAGES},
                     "documents": count, "percent_of_baseline": safe_percentage(count, total)})
    return rows


def overlap_summary(patterns, iterations=2000, seed=42):
    pattern_counts = Counter(patterns)
    total = len(patterns)
    intervals = bootstrap(pattern_counts, iterations, seed)
    pairs = []
    for a, b in PAIRS:
        row = pair_statistics(pattern_counts, a, b)
        for metric in METRICS:
            low, high = intervals.get((a, b, metric), (None, None))
            row[metric + "_ci95_low"], row[metric + "_ci95_high"] = low, high
        pairs.append(row)
    per_filter = []
    for stage in STAGES:
        removed = removed_by(pattern_counts, stage)
        exclusive = pattern_counts.get(1 << STAGES.index(stage), 0)
        per_filter.append({"filter": stage, "removed": removed,
                           "removed_percent": safe_percentage(removed, total),
                           "removed_only_by_this_filter": exclusive,
                           "exclusive_share_of_own_removals": exclusive / removed if removed else None,
                           "also_removed_by_another_filter": removed - exclusive})
    by_k = Counter()
    for pattern, count in pattern_counts.items():
        by_k[bin(pattern).count("1")] += count
    return {"documents": total, "pattern_counts": pattern_rows(pattern_counts, total),
            "removed_by_exactly": {str(k): by_k.get(k, 0) for k in range(len(STAGES) + 1)},
            "survived_all_filters": pattern_counts.get(0, 0),
            "removed_by_any_filter": total - pattern_counts.get(0, 0),
            "per_filter": per_filter, "pairs": pairs,
            "bootstrap": {"iterations": iterations, "seed": seed, "method":
                          "percentile; documents resampled with replacement (multinomial over removal patterns)"}}


# ---------------------------------------------------------------------------
# Removed-set structural profiles


class Profile:
    def __init__(self):
        self.documents, self.tokens, self.characters = 0, 0, 0
        self.domains, self.sentences, self.document_tokens = Counter(), Counter(), Counter()

    def add(self, domain, lengths, characters):
        self.documents += 1
        self.tokens += sum(lengths)
        self.characters += characters
        self.domains[domain] += 1
        self.sentences.update(lengths)
        self.document_tokens[sum(lengths)] += 1

    def summarize(self):
        return {"documents": self.documents, "tokens": self.tokens, "characters": self.characters,
                "document_tokens_median": histogram_quantile(self.document_tokens, .5),
                "document_tokens_p25": histogram_quantile(self.document_tokens, .25),
                "document_tokens_p75": histogram_quantile(self.document_tokens, .75),
                **sentence_statistics(self.sentences),
                "unique_domains": len(self.domains), "domain_entropy_bits": entropy_bits(self.domains),
                "top_domain_share": (max(self.domains.values()) / self.documents) if self.documents else None}


def wasserstein(first, second):
    """1-Wasserstein distance between two integer-valued histograms."""
    if not sum(first.values()) or not sum(second.values()):
        return None
    support = sorted(set(first) | set(second))
    total_a, total_b = sum(first.values()), sum(second.values())
    cumulative_a = cumulative_b = distance = 0.0
    for value, following in zip(support, support[1:]):
        cumulative_a += first.get(value, 0) / total_a
        cumulative_b += second.get(value, 0) / total_b
        distance += abs(cumulative_a - cumulative_b) * (following - value)
    return distance


def profile_groups(pattern):
    groups = ["baseline"]
    if pattern == 0:
        groups.append("survived_all")
    for bit, stage in enumerate(STAGES):
        if pattern & (1 << bit):
            groups.append(f"{stage}:removed")
            if pattern == 1 << bit:
                groups.append(f"{stage}:removed_only_here")
        else:
            groups.append(f"{stage}:retained")
    return groups


def removed_profiles(loaded, sentence_lengths):
    baseline = loaded["baseline"]
    before = sha256_file(baseline)
    if before != loaded["baseline_sha256"]:
        raise ValueError(f"{loaded['dataset']}: local baseline differs from the run's recorded baseline")
    profiles = {}
    index = -1
    for index, (_, document) in enumerate(baseline_rows(baseline, 20000)):
        if index >= len(loaded["ids"]) or document["document_id"] != loaded["ids"][index]:
            raise ValueError(f"{loaded['dataset']}: baseline order differs from decision logs at row {index + 1}")
        lengths = sentence_lengths(document["text"])
        for group in profile_groups(loaded["patterns"][index]):
            profiles.setdefault(group, Profile()).add(document["domain"], lengths, document["character_count"])
    if index + 1 != len(loaded["ids"]) or sha256_file(baseline) != before:
        raise RuntimeError(f"{loaded['dataset']}: baseline changed or is incomplete")
    base = profiles["baseline"]
    base_summary = base.summarize()
    rows = []
    for group in ["baseline", "survived_all"] + [f"{s}:{kind}" for s in STAGES
                                                for kind in ("removed", "removed_only_here", "retained")]:
        profile = profiles.get(group, Profile())
        stage, _, kind = group.partition(":")
        complement = profiles.get(f"{stage}:retained") if kind == "removed" else None
        row = {"group": group, "filter": stage if kind else None, "set": kind or group, **profile.summarize(),
               "sentence_wasserstein_vs_baseline": wasserstein(profile.sentences, base.sentences),
               "sentence_wasserstein_vs_retained": (wasserstein(profile.sentences, complement.sentences)
                                                    if complement else None)}
        for key in ("sentence_mean_tokens", "sentence_median_tokens", "domain_entropy_bits", "document_tokens_median"):
            value, original = row[key], base_summary[key]
            row[key + "_delta_vs_baseline"] = value - original if value is not None and original is not None else None
        rows.append(row)
    histograms = {group: {str(k): v for k, v in sorted(profile.sentences.items())} for group, profile in profiles.items()}
    return rows, histograms


# ---------------------------------------------------------------------------
# Figures


def _plt():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def heatmap_figure(summary, name, destination):
    plt = _plt()
    import numpy as np

    size = len(STAGES)
    jaccard, containment, phi = (np.full((size, size), np.nan) for _ in range(3))
    for row in summary["pairs"]:
        i, j = STAGES.index(row["filter_a"]), STAGES.index(row["filter_b"])
        for matrix, a_b, b_a in ((jaccard, "jaccard", "jaccard"), (phi, "phi", "phi"),
                                 (containment, "containment_a_in_b", "containment_b_in_a")):
            matrix[i, j] = np.nan if row[a_b] is None else row[a_b]
            matrix[j, i] = np.nan if row[b_a] is None else row[b_a]
    labels = [LABELS[s] for s in STAGES]
    panels = ((jaccard, "Jaccard of removed sets", "Blues", (0, 1)),
              (containment, "Containment: share of row's removals\nalso removed by column", "Blues", (0, 1)),
              (phi, "Phi (chance-corrected association)", "RdBu_r", (-1, 1)))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        for axis, (matrix, title, cmap, limits) in zip(axes, panels):
            image = axis.imshow(matrix, cmap=cmap, vmin=limits[0], vmax=limits[1])
            axis.set_xticks(range(size), labels)
            axis.set_yticks(range(size), labels)
            axis.set_title(title)
            for i in range(size):
                for j in range(size):
                    if i == j:
                        axis.text(j, i, "—", ha="center", va="center", color="#52514e")
                    elif not np.isnan(matrix[i, j]):
                        value = matrix[i, j]
                        dark = abs(value) > (.6 if limits[0] == 0 else .5)
                        axis.text(j, i, f"{value:.2f}", ha="center", va="center",
                                  color="white" if dark else "#0b0b0b")
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
            figure.colorbar(image, ax=axis, fraction=.046, pad=.04)
        figure.suptitle(f"{name}: cross-filter overlap of removed documents (n = {summary['documents']:,})")
        figure.tight_layout()
        figure.savefig(destination, dpi=160, metadata={"Software": "FineWeb Filtering Research"})
        plt.close(figure)


def upset_figure(summary, name, destination):
    plt = _plt()
    rows = sorted((row for row in summary["pattern_counts"] if row["documents"]), key=lambda r: -r["documents"])
    total = summary["documents"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure, (bars, dots) = plt.subplots(2, 1, figsize=(max(7, .62 * len(rows) + 3), 6.2),
                                            gridspec_kw={"height_ratios": [3, 1.3], "hspace": .05}, sharex=True)
        x = range(len(rows))
        bars.bar(x, [r["documents"] for r in rows], color="#2a78d6", width=.62)
        for index, row in enumerate(rows):
            bars.text(index, row["documents"], f"{row['documents']:,}\n{100 * row['documents'] / total:.1f}%",
                      ha="center", va="bottom", fontsize=8, color="#0b0b0b")
        bars.set_ylabel("Documents")
        bars.set_ylim(0, max(r["documents"] for r in rows) * 1.22)
        bars.spines[["top", "right"]].set_visible(False)
        bars.grid(axis="y", color="#e4e3df", linewidth=.6)
        bars.set_axisbelow(True)
        bars.set_title(f"{name}: documents by exact combination of removing filters")
        for level, stage in enumerate(STAGES):
            for index, row in enumerate(rows):
                active = bool(row[stage])
                dots.scatter(index, level, s=70, color="#0b0b0b" if active else "#d9d8d3", zorder=3)
            dots.axhline(level, color="#f0efec", linewidth=6, zorder=1)
        for index, row in enumerate(rows):
            levels = [level for level, stage in enumerate(STAGES) if row[stage]]
            if len(levels) > 1:
                dots.plot([index, index], [min(levels), max(levels)], color="#0b0b0b", linewidth=2, zorder=2)
        dots.set_yticks(range(len(STAGES)), [f"{LABELS[s]} removes" for s in STAGES])
        dots.set_xticks(list(x), ["" for _ in rows])
        dots.set_ylim(-.6, len(STAGES) - .4)
        dots.invert_yaxis()
        for spine in dots.spines.values():
            spine.set_visible(False)
        dots.tick_params(length=0)
        figure.text(.5, .01, "Column with no black dots = documents every filter keeps.", ha="center", fontsize=9,
                    color="#52514e")
        figure.savefig(destination, dpi=160, bbox_inches="tight", metadata={"Software": "FineWeb Filtering Research"})
        plt.close(figure)


def cdf_figure(histograms, name, destination):
    plt = _plt()
    import numpy as np

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure, axis = plt.subplots(figsize=(8.5, 5))
        series = [("baseline", "Baseline", COLORS["baseline"], "--")] + [
            (f"{s}:removed", f"{LABELS[s]} removed", COLORS[s], "-") for s in STAGES]
        for group, label, color, style in series:
            histogram = {int(k): v for k, v in histograms.get(group, {}).items()}
            if not histogram:
                continue
            x = np.array(sorted(histogram), dtype=float)
            y = np.cumsum([histogram[int(v)] for v in x]) / sum(histogram.values())
            axis.step(np.r_[x[0], x], np.r_[0, y], where="post", color=color, linestyle=style,
                      linewidth=2, label=label)
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("Sentence length (tokens, log scale)")
        axis.set_ylabel("Cumulative share of sentences")
        axis.set_ylim(0, 1.01)
        axis.grid(color="#e4e3df", linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="lower right")
        axis.set_title(f"{name}: sentence lengths of each filter's removed documents")
        figure.tight_layout()
        figure.savefig(destination, dpi=160, metadata={"Software": "FineWeb Filtering Research"})
        plt.close(figure)


# ---------------------------------------------------------------------------
# Orchestration


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def default_sentence_lengths(assets_directory):
    from .metrics.calculations import MeasurementTokenizer
    return MeasurementTokenizer(Path(assets_directory)).sentence_lengths


def analyze(runs, output=ROOT / "results", iterations=2000, seed=42, profiles=True, sentence_lengths=None):
    """Analyze one or more independent runs into a single new dated folder."""
    output = Path(output).resolve()
    loaded = [load_run(run) for run in runs]
    if not loaded:
        raise ValueError("Provide at least one independent run")
    names = [entry["dataset"] for entry in loaded]
    if len(set(names)) != len(names):
        raise ValueError("Analyze at most one run per dataset in a single overlap folder")
    started = now()
    directory = create_run(output, "cross_filter", started, mode="overlap")
    combined_pairs, combined_filters, combined_profiles, record = [], [], [], {}
    for entry in loaded:
        name = entry["dataset"]
        target = directory / name
        target.mkdir()
        summary = overlap_summary(entry["patterns"], iterations, seed)
        write_csv(target / "pairs.csv", summary["pairs"])
        write_csv(target / "patterns.csv", summary["pattern_counts"])
        write_csv(target / "per_filter.csv", summary["per_filter"])
        heatmap_figure(summary, name, target / "overlap_heatmaps.png")
        upset_figure(summary, name, target / "removal_patterns.png")
        if profiles:
            measure = sentence_lengths or default_sentence_lengths(entry["assets_directory"])
            rows, histograms = removed_profiles(entry, measure)
            summary["profiles"] = rows
            write_csv(target / "profiles.csv", rows)
            write_json(target / "sentence_histograms.json", histograms)
            cdf_figure(histograms, name, target / "removed_sentence_cdf.png")
            combined_profiles += [{"dataset": name, **row} for row in rows]
        summary["source"] = {"run": str(entry["run"]), "baseline": str(entry["baseline"]),
                             "baseline_sha256": entry["baseline_sha256"], "filters": entry["sources"]}
        write_json(target / "overlap.json", summary)
        combined_pairs += [{"dataset": name, **row} for row in summary["pairs"]]
        combined_filters += [{"dataset": name, **row} for row in summary["per_filter"]]
        record[name] = {"documents": summary["documents"], "survived_all_filters": summary["survived_all_filters"],
                        "removed_by_exactly": summary["removed_by_exactly"], "folder": name}
    write_csv(directory / "pairs_all_datasets.csv", combined_pairs)
    write_csv(directory / "per_filter_all_datasets.csv", combined_filters)
    if combined_profiles:
        write_csv(directory / "profiles_all_datasets.csv", combined_profiles)
    write_json(directory / "summary.json", {
        "status": "complete", "analysis": "cross_filter_overlap", "started_at": started.isoformat(),
        "completed_at": now().isoformat(), "timezone": str(LOCAL_TIME), "datasets": record,
        "source_runs": [str(entry["run"]) for entry in loaded], "removed_set_profiles": profiles,
        "bootstrap_iterations": iterations, "seed": seed,
        "interpretation_notes": [
            "All filters judged the same baseline independently; overlap is not a sequential-pipeline effect.",
            "Deduplication decisions depend on the whole input; its overlap reflects which copy was kept.",
            "Containment and Jaccard rise mechanically with removal rate; compare against lift and phi.",
            "Confidence intervals describe resampling of this bounded sample, not Common Crawl as a whole."]})
    print(f"Saved cross-filter overlap analysis: {directory}", flush=True)
    return directory


def latest_independent_runs(results=ROOT / "results"):
    """Newest complete independent run per dataset (folder names sort chronologically)."""
    chosen = {}
    for summary_path in sorted(Path(results).glob("*_independent_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete" and summary.get("filter_mode") == "independent":
            chosen[summary["dataset"]] = summary_path.parent
    return [chosen[name] for name in sorted(chosen)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", nargs="+", type=Path, help="Completed independent run folders")
    source.add_argument("--latest", action="store_true", help="Use the newest complete independent run per dataset")
    parser.add_argument("--iterations", type=int, default=2000, help="Bootstrap resamples (0 disables intervals)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-profiles", action="store_true",
                        help="Skip removed-set text profiles (no tokenizer or baseline text needed)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    runs = args.runs or latest_independent_runs(args.output_dir)
    if not runs:
        parser.error("No complete independent runs found; run independent filtering first")
    analyze(runs, args.output_dir, args.iterations, args.seed, profiles=not args.no_profiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
