"""Descriptive language, overlap and filtering tables; no hypothesis evaluation."""

import csv
from collections import Counter, defaultdict
from itertools import combinations
import json

from src.filter_assets import ROOT, sha256_file
from src.filter_text import simplify_text
from src.language_filter import LanguageFilter
from src.run_filters import STAGES, load_config
from .common import EVIDENCE, baseline_path, check_legacy, ledger_rows, run_root, write_json
from .prepare import replay_selection, sampling_config
from .run import validate_stage


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def pair_overlap(first, second):
    """Report overlap; do not delete or replace any selected document."""
    rows = []
    for metric, key in (
            ("document_id", lambda row: row["document_id"]),
            ("warc_record_id", lambda row: row["source"]["warc_record_id"]),
            ("url", lambda row: row["url"]),
            ("exact_capped_text", lambda row: row["exact_capped_text"]),
            ("normalized_capped_text", lambda row: row["normalized_capped_text"])):
        a, b = Counter(map(key, first)), Counter(map(key, second))
        shared = set(a) & set(b)
        rows.append({"metric": metric, "shared_unique_values": len(shared),
                     "matching_record_pairs": sum(a[value] * b[value] for value in shared),
                     "matching_documents_a": sum(a[value] for value in shared),
                     "matching_documents_b": sum(b[value] for value in shared)})
    return rows


def sha256_file_content(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprints(row):
    return {"document_id": row["document_id"], "url": row["url"], "source": row["source"],
            "exact_capped_text": sha256_file_content(row["text"]),
            "text_characters": len(row["text"]),
            "normalized_capped_text": sha256_file_content(simplify_text(row["text"]))}


def summarize():
    protected_count = check_legacy()
    config = load_config(ROOT / "filter_config.json")
    model = LanguageFilter(config["language"], ROOT / config["assets_directory"])
    language_rows, language_summaries, filter_rows, inventories = [], [], [], {}
    baseline_hashes = {}
    for dataset in (1, 2, 3):
        path = baseline_path(dataset)
        before = sha256_file(path)
        baseline_hashes[str(dataset)] = before
        counts, low, prediction_rows = Counter(), Counter(), []
        source_rows = []
        for row in ledger_rows(path):
            accepted, reason, details = model.evaluate(row)
            language = details["language"] or "und"
            low_confidence = details["confidence"] is not None and details["confidence"] <= config["language"]["english_confidence_threshold"]
            counts[language] += 1
            low[language] += low_confidence
            prediction_rows.append({"document_id": row["document_id"], **details, "low_confidence": low_confidence,
                                    "language_gate_kept": accepted, "gate_reason": reason})
            source_rows.append(fingerprints(row))
        assert sum(counts.values()) == 1000
        assert sha256_file(path) == before
        inventories[dataset] = source_rows
        prediction_path = EVIDENCE / f"dataset_{dataset}/language_predictions.jsonl"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
            for prediction in prediction_rows:
                stream.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
        for language, count in counts.most_common():
            language_rows.append({"dataset": dataset, "language": language, "documents": count,
                                  "percentage": count / 10, "low_confidence_documents": low[language]})
        language_summaries.append({"dataset": dataset, "documents": 1000, "english_predicted": counts["en"],
                                   "non_english_predicted": 1000 - counts["en"] - counts["und"],
                                   "unclassified": counts["und"], "all_low_confidence": sum(low.values()),
                                   "english_low_confidence": low["en"],
                                   "english_above_threshold": counts["en"] - low["en"],
                                   "confidence_threshold": config["language"]["english_confidence_threshold"]})
        root = ROOT if dataset == 1 else run_root(dataset)
        saved_language = {row["document_id"]: row for row in ledger_rows(root / "results/language_decisions.jsonl")}
        for prediction in prediction_rows:
            saved = saved_language[prediction["document_id"]]
            assert prediction["language"] == saved["details"]["language"]
            assert prediction["confidence"] == saved["details"]["confidence"]
            assert prediction["language_gate_kept"] == saved["kept"]
        for stage in STAGES:
            summary = json.loads((root / f"results/{stage}_summary.json").read_text(encoding="utf-8"))
            assert summary["baseline_sha256"] == before == summary["baseline_sha256_after"]
            assert all(summary["config"][key] == config[key] for key in (*STAGES, "max_text_characters", "expected_documents"))
            filter_rows.append({"dataset": dataset, "filter": stage, "input": summary["input_count"],
                                "retained": summary["output_count"], "removed": summary["removed_count"],
                                "removed_percentage": round(summary["removed_count"] / summary["input_count"] * 100, 1)})
        print(f"Dataset {dataset}: {counts['en']} predicted English; {sum(low.values())} low-confidence predictions across all languages", flush=True)
    overlap = []
    exact_details = []
    for first, second in combinations((1, 2, 3), 2):
        overlap.extend({"dataset_a": first, "dataset_b": second, **row} for row in pair_overlap(inventories[first], inventories[second]))
        a = Counter(row["exact_capped_text"] for row in inventories[first])
        b = Counter(row["exact_capped_text"] for row in inventories[second])
        lengths = {row["exact_capped_text"]: row["text_characters"] for row in inventories[first]}
        for fingerprint in sorted(set(a) & set(b)):
            exact_details.append({"dataset_a": first, "dataset_b": second, "text_sha256": fingerprint,
                                  "text_characters": lengths[fingerprint], "documents_a": a[fingerprint],
                                  "documents_b": b[fingerprint], "matching_record_pairs": a[fingerprint] * b[fingerprint]})
    write_csv(EVIDENCE / "language_distribution.csv", ["dataset", "language", "documents", "percentage", "low_confidence_documents"], language_rows)
    write_csv(EVIDENCE / "language_summary.csv", list(language_summaries[0]), language_summaries)
    write_csv(EVIDENCE / "filtering_results.csv", list(filter_rows[0]), filter_rows)
    write_csv(EVIDENCE / "cross_dataset_overlap.csv", list(overlap[0]), overlap)
    write_csv(EVIDENCE / "shared_text_details.csv", ["dataset_a", "dataset_b", "text_sha256", "text_characters",
                                                   "documents_a", "documents_b", "matching_record_pairs"], exact_details)
    selection = [replay_selection(dataset, sampling_config()) for dataset in (2, 3)]
    for dataset in (1, 2, 3):
        assert sha256_file(baseline_path(dataset)) == baseline_hashes[str(dataset)]
    assert check_legacy() == protected_count
    evidence = {"language_summary": language_summaries, "filtering_results": filter_rows, "overlap": overlap,
                "baseline_sha256": baseline_hashes, "legacy_files_unchanged": protected_count,
                "exact_text_overlap_details": exact_details,
                "sampling_replay": selection, "language_predictions_match_stage_ledgers": True,
                "same_thresholds_for_all_datasets": True}
    write_json(EVIDENCE / "cross_dataset_validation.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    summarize()
