"""Dated filtering runs, JSON exports and document-retention charts."""

import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re

from .filter_assets import ROOT, sha256_file
from .run_filters import baseline_rows

LOCAL_TIME = timezone(timedelta(hours=5), "Asia/Qyzylorda")
LABELS = {"url": "URL", "language": "Language", "quality": "Quality", "dedup": "Deduplication"}


def now():
    return datetime.now(LOCAL_TIME)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inspect_baseline(path, limit):
    before = sha256_file(path)
    count, crawls = 0, set()
    for _, document in baseline_rows(path, limit):
        count += 1
        match = re.search(r"CC-MAIN-(\d{4})-(\d{2})", document["source"]["wet_path"])
        if match:
            crawls.add("CC_" + "_".join(match.groups()))
    if not count:
        raise ValueError("The baseline must contain at least one document")
    if sha256_file(path) != before:
        raise RuntimeError("Baseline changed during validation")
    return count, before, crawls


def create_run(parent, name, started):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Dataset names may contain only letters, digits, underscores and hyphens")
    if started.tzinfo is None:
        raise ValueError("Run timestamps must include a timezone")
    stamp = started.astimezone(LOCAL_TIME).strftime("%Y-%m-%d")
    clock = started.astimezone(LOCAL_TIME).strftime("%H-%M-%S")
    parent.mkdir(parents=True, exist_ok=True)
    base = f"{stamp}_{name}_{clock}"
    for suffix in range(10000):
        directory = parent / (base if suffix == 0 else f"{base}_{suffix}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique results directory")


def export_json(source, destination):
    partial = destination.with_suffix(".json.partial")
    if destination.exists():
        raise FileExistsError(destination)
    count = 0
    with source.open(encoding="utf-8") as stream, partial.open("x", encoding="utf-8", newline="\n") as target:
        target.write("[\n")
        for line in stream:
            document = json.loads(line)
            if count:
                target.write(",\n")
            target.write(json.dumps(document, ensure_ascii=False, allow_nan=False))
            count += 1
        target.write("\n]\n")
    partial.replace(destination)
    return count


def retention_chart(rows, destination, name):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    counts = [rows[0]["input"]] + [row["retained"] for row in rows]
    removed = [0] + [row["removed"] for row in rows]
    labels = ["Baseline"] + [LABELS[row["filter"]] for row in rows]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        figure, axis = plt.subplots(figsize=(9, 5.2))
        x = list(range(len(counts)))
        axis.bar(x, counts, color=["#52677f"] + ["#16856b"] * len(rows), label="Retained")
        axis.bar(x, removed, bottom=counts, color="#e6b7ad", label="Removed")
        for index, count in enumerate(counts):
            percent = count / counts[0] * 100
            axis.text(index, counts[0] * 1.025, f"{count:,}\n({percent:.1f}%)", ha="center", va="bottom")
        axis.set_xticks(x, labels)
        axis.set_ylim(0, counts[0] * 1.25)
        axis.set_ylabel("Documents")
        axis.set_title(f"{name}: documents retained by independent filters")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(handles=[Patch(color="#16856b", label="Retained"), Patch(color="#e6b7ad", label="Removed")],
                    loc="upper center", bbox_to_anchor=(.5, -.10), ncol=2, frameon=False)
        figure.text(.5, .015, "Every filter receives the original baseline. Filters are not sequential.", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, .08, 1, 1))
        figure.savefig(destination, dpi=160, metadata={"Software": "FineWeb Filtering Research"})
        plt.close(figure)


def publish(directory, name, baseline, baseline_hash, started, outputs, origin, completed=None):
    datasets = directory / "datasets"
    datasets.mkdir()
    rows, sources = [], {}
    for stage, files in outputs.items():
        report = json.loads(files["summary"].read_text(encoding="utf-8"))
        if (report["status"] != "complete" or report["baseline_sha256"] != baseline_hash
                or report["baseline_sha256_after"] != baseline_hash
                or sha256_file(files["dataset"]) != report["output_sha256"]):
            raise ValueError(f"Source verification failed for {stage}")
        destination = datasets / f"{stage}.json"
        count = export_json(files["dataset"], destination)
        if count != report["output_count"] or report["input_count"] != count + report["removed_count"]:
            raise ValueError(f"Document counts do not reconcile for {stage}")
        rows.append({"filter": stage, "input": report["input_count"], "retained": count,
                     "removed": report["removed_count"], "retained_percent": 100 * count / report["input_count"],
                     "removed_percent": 100 * report["removed_count"] / report["input_count"]})
        sources[stage] = {"source_summary": str(files["summary"].resolve()),
                          "source_jsonl": str(files["dataset"].resolve()), "source_jsonl_sha256": report["output_sha256"],
                          "json": f"datasets/{stage}.json", "json_sha256": sha256_file(destination),
                          "removal_reasons": report["removal_reasons"]}
    if not rows or len({row["input"] for row in rows}) != 1:
        raise ValueError("All conditions must use the same nonempty baseline")
    if sha256_file(baseline) != baseline_hash:
        raise RuntimeError("Baseline changed; results are incomplete")
    with (directory / "retention.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    retention_chart(rows, directory / "retention.png", name)
    summary = {"status": "complete", "dataset": name, "started_at": started.astimezone(LOCAL_TIME).isoformat(),
               "completed_at": (completed or now()).astimezone(LOCAL_TIME).isoformat(),
               "packaged_at": now().isoformat(), "timezone": "Asia/Qyzylorda (UTC+05:00)", "origin": origin,
               "filter_mode": "independent", "baseline": str(baseline.resolve()), "baseline_sha256": baseline_hash,
               "baseline_documents": rows[0]["input"], "filters": rows, "artifacts": sources,
               "quality_text": "Retained original text; C4 working-text transformations are not exported"}
    write_json(directory / "summary.json", summary)
    return summary
