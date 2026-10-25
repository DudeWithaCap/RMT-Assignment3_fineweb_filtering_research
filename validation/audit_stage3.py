"""Audit filtering decisions and pinned upstream rule parity."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from src.filter_assets import DATATROVE_REVISION, ROOT, sha256_file
from src.filter_text import PUNCTUATION_SET
from src.filter_tokenizer import EnglishTokenizer
from src.run_filters import STAGES, baseline_rows, load_config
from src.quality_filter import QualityFilter
from .reference_quality import REFERENCE_FILES, build_reference, reference_decision

AUDIT_SEED = "stage3-validation-v1"


def read_jsonl(path):
    # Iterate physical JSONL lines: Unicode separators can occur inside strings.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def rank(document_id):
    return hashlib.sha256(f"{AUDIT_SEED}:{document_id}".encode()).hexdigest()


def review_sample(ledger):
    strata = defaultdict(list)
    for record in ledger:
        group = "kept_english" if record["kept"] else (
            "low_confidence_english" if record["reason"] == "language_low_confidence" else "predicted_non_english")
        strata[group].append(record)
    selected = []
    for group, records in sorted(strata.items()):
        for record in sorted(records, key=lambda record: rank(record["document_id"]))[:10]:
            selected.append({"document_id": record["document_id"], "stratum": group})
    return sorted(selected, key=lambda record: rank("blind:" + record["document_id"]))


def fetch_reference(directory):
    import requests

    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/huggingface/datatrove/{DATATROVE_REVISION}/"
    for name in REFERENCE_FILES:
        relative = "examples/fineweb.py" if name == "fineweb.py" else (
            "src/datatrove/utils/text.py" if name == "text.py" else "src/datatrove/pipeline/filters/" + name)
        url = base + relative
        response = requests.get(url, timeout=(15, 60))
        response.raise_for_status()
        if len(response.content) > 1024 * 1024:
            raise ValueError("Unexpected reference source size")
        (directory / name).write_bytes(response.content)


def audit(fetch=False, print_review=False):
    config = load_config(ROOT / "filter_config.json")
    baseline = ROOT / config["baseline"]
    protected = [baseline, ROOT / "filter_config.json"] + [ROOT / f"results/{stage}_{suffix}" for stage in STAGES
                 for suffix in ("summary.json", "decisions.jsonl", "removed.jsonl")] + [
                     ROOT / f"data/processed/filtered_{stage}.jsonl" for stage in STAGES]
    before = {str(path.relative_to(ROOT)): sha256_file(path) for path in protected}
    rows = [document for _, document in baseline_rows(baseline, config["max_text_characters"])]
    documents = {row["document_id"]: row for row in rows}
    assert len(rows) == config["expected_documents"]
    assert sha256_file(baseline) == config["baseline_sha256"]
    ledgers, summaries = {}, {}
    for stage in STAGES:
        ledger = read_jsonl(ROOT / f"results/{stage}_decisions.jsonl")
        summary = json.loads((ROOT / f"results/{stage}_summary.json").read_text(encoding="utf-8"))
        retained = read_jsonl(ROOT / f"data/processed/filtered_{stage}.jsonl")
        assert len(ledger) == len(rows)
        assert {r["document_id"] for r in ledger} == set(documents)
        assert summary["baseline_sha256"] == config["baseline_sha256"] == summary["baseline_sha256_after"]
        assert all(row == documents[row["document_id"]] for row in retained)
        assert sha256_file(ROOT / f"data/processed/filtered_{stage}.jsonl") == summary["output_sha256"]
        assert sha256_file(ROOT / f"results/{stage}_decisions.jsonl") == summary["decisions_sha256"]
        ledgers[stage] = ledger
        summaries[stage] = summary
    assert len({summary["input_ids_sha256"] for summary in summaries.values()}) == 1
    output = ROOT / "results/validation"
    output.mkdir(parents=True, exist_ok=True)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "baseline_sha256": config["baseline_sha256"],
              "filter_config_sha256": before["filter_config.json"], "input_count": len(rows),
              "stage_counts": {stage: {key: summary[key] for key in ("input_count", "output_count", "removed_count", "removal_reasons")}
                               for stage, summary in summaries.items()}}
    languages = Counter(record["details"]["language"] for record in ledgers["language"])
    result["predicted_language_distribution"] = dict(languages.most_common())
    result["language_rejection_reasons"] = dict(Counter(record["reason"] for record in ledgers["language"] if not record["kept"]))
    result["english_confidence_bins"] = dict(Counter(
        ">0.65" if record["details"]["confidence"] > .65 else "0.50-0.65" if record["details"]["confidence"] >= .5
        else "<0.50" for record in ledgers["language"] if record["details"]["language"] == "en"))
    quality_by_id = {record["document_id"]: record for record in ledgers["quality"]}
    cross = defaultdict(Counter)
    for record in ledgers["language"]:
        group = "confident_english" if record["kept"] else (
            "low_confidence_english" if record["details"]["language"] == "en" else "predicted_non_english")
        quality = quality_by_id[record["document_id"]]
        cross[group]["total"] += 1
        cross[group]["quality_kept" if quality["kept"] else "quality_removed"] += 1
        if quality["reason"]:
            cross[group][quality["reason"]] += 1
    result["quality_by_language_diagnostic_only"] = {key: dict(value) for key, value in cross.items()}
    result["sampling"] = {
        "distinct_wet_files": len({row["source"]["wet_path"] for row in rows}),
        "first_record_index": rows[0]["source"]["record_index"],
        "last_record_index": rows[-1]["source"]["record_index"],
        "unique_hostnames": len({row["domain"] for row in rows}),
        "first_hostname": rows[0]["domain"], "last_hostname": rows[-1]["domain"],
        "truncated_documents": sum(row["truncated"] for row in rows),
        "crawl_dates": dict(Counter((row["source"].get("warc_date") or "unknown")[:10] for row in rows)),
    }
    tokenizer = EnglishTokenizer(ROOT / config["assets_directory"])
    reference_directory = ROOT / config["assets_directory"] / "validation_reference"
    if fetch:
        fetch_reference(reference_directory)
    filters, namespace = build_reference(reference_directory, tokenizer)
    assert namespace["PUNCTUATION_SET"] == PUNCTUATION_SET
    current = QualityFilter(config["quality"], tokenizer)
    mismatches, current_log_mismatches, reference_failures, errors = [], [], Counter(), []
    cleaned_changes, cleaned_removed_characters = 0, 0
    for row in rows:
        try:
            accepted, component, reason = reference_decision(filters, row["text"])
            actual, actual_reason, details = current.evaluate(row)
            if accepted != actual:
                mismatches.append({"document_id": row["document_id"], "reference": accepted, "current": actual,
                                   "reference_component": component, "reference_reason": reason, "current_reason": actual_reason})
            logged = quality_by_id[row["document_id"]]
            if (actual, actual_reason) != (logged["kept"], logged["reason"]):
                current_log_mismatches.append(row["document_id"])
            if not accepted:
                reference_failures[f"{component}:{reason}"] += 1
            if accepted and details["c4_working_characters"] != len(row["text"]):
                cleaned_changes += 1
                cleaned_removed_characters += len(row["text"]) - details["c4_working_characters"]
        except Exception as error:
            errors.append({"document_id": row["document_id"], "error": f"{type(error).__name__}: {error}"})
    result["reference_quality_parity"] = {
        "datatrove_revision": DATATROVE_REVISION, "checked": len(rows), "errors": errors,
        "decision_mismatches": mismatches, "current_vs_saved_log_mismatches": current_log_mismatches,
        "first_failures": dict(reference_failures.most_common()),
        "sources_sha256": {name: sha256_file(reference_directory / name) for name in REFERENCE_FILES},
        "limitation": "Same installed tokenizer in both implementations; checks rule bodies/defaults, not 2024 tokenizer or extraction equivalence",
        "accepted_documents_changed_by_reference_c4": cleaned_changes,
        "reference_c4_characters_removed_from_accepted": cleaned_removed_characters,
    }
    # Supplementary full-input diagnostics for the three largest first-failure
    # causes. These overlap; they do not replace the production decision order.
    diagnostic = Counter()
    for row in rows:
        words = tokenizer.word_tokenize(row["text"])
        content = [word for word in words if any(char not in PUNCTUATION_SET for char in word)]
        if words and sum(any(c.isalpha() for c in word) for word in words) / len(words) < config["quality"]["min_alpha_word_ratio"]:
            diagnostic["below_alpha_ratio_on_full_baseline"] += 1
        if sum(word in config["quality"]["stop_words"] for word in words) < config["quality"]["min_stop_words"]:
            diagnostic["insufficient_english_stop_words_on_full_baseline"] += 1
        if content and sum(map(len, content)) / len(content) > config["quality"]["max_avg_word_length"]:
            diagnostic["long_average_words_on_full_baseline"] += 1
    result["overlapping_rule_diagnostics"] = dict(diagnostic)
    sample = review_sample(ledgers["language"])
    (output / "language_sample_selection.json").write_text(json.dumps({"seed": AUDIT_SEED, "method": "10 lowest SHA256(seed:id) from each model-decision stratum, re-ranked for blinded display", "sample": sample}, indent=2) + "\n", encoding="utf-8")
    if print_review:
        for index, selection in enumerate(sample, start=1):
            row = documents[selection["document_id"]]
            text = row["text"]
            middle = len(text) // 2
            excerpt = text if len(text) <= 1600 else text[:800] + "\n[MIDDLE]\n" + text[middle:middle + 400] + "\n[END]\n" + text[-400:]
            print(f"REVIEW {index} {row['document_id']} characters={len(text)}\n{excerpt}\n", flush=True)
    after = {str(path.relative_to(ROOT)): sha256_file(path) for path in protected}
    assert before == after, "Protected baseline/config/filter artifacts changed during validation"
    result["protected_inputs_and_outputs_unchanged"] = True
    result["protected_file_sha256"] = before
    (output / "audit_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("protected_file_sha256",)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-reference", action="store_true")
    parser.add_argument("--print-review", action="store_true")
    arguments = parser.parse_args()
    audit(arguments.fetch_reference, arguments.print_review)
