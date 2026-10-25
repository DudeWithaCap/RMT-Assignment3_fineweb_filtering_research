"""Measure existing conditions sequentially; never run or change primary filters."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace

from src.filter_assets import ROOT, sha256_file
from src.run_filters import STAGES, baseline_rows, load_config
from stage35.common import baseline_path, run_root
from validation.reference_quality import build_reference
from .metrics import Accumulator, MeasurementTokenizer, relative_to_baseline, safe_percentage
from .preserve import OUTPUT, verify_preserved

CONDITIONS = ("baseline", *STAGES)


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("A table must have a schema row")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_inputs(dataset):
    root = ROOT if dataset == 1 else run_root(dataset)
    metadata_path = ROOT / "results/dataset_metadata.json" if dataset == 1 else ROOT / f"results/stage35/dataset_{dataset}/preparation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    baseline_hash = metadata["dataset_sha256"]
    original = load_config(ROOT / "filter_config.json")
    result = {"baseline": {"path": baseline_path(dataset), "sha256": baseline_hash, "count": 1000}}
    for stage in STAGES:
        summary_path = root / f"results/{stage}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["status"] != "complete" or summary["baseline_sha256"] != baseline_hash or summary["baseline_sha256_after"] != baseline_hash:
            raise ValueError(f"Wrong baseline or incomplete {dataset}/{stage} input")
        if any(summary["config"][key] != original[key] for key in (*STAGES, "max_text_characters")):
            raise ValueError("Existing condition configurations differ")
        result[stage] = {"path": root / f"data/processed/filtered_{stage}.jsonl", "sha256": summary["output_sha256"],
                         "count": summary["output_count"], "summary_path": summary_path,
                         "input_ids_sha256": summary["input_ids_sha256"]}
    return result


def c4_transformer(tokenizer, config):
    directory = ROOT / "data/assets/validation_reference"
    _, namespace = build_reference(directory, tokenizer.tokenizer)
    quality = config["quality"]
    transformer = namespace["C4QualityFilter"](
        filter_no_terminal_punct=False, remove_citations=quality["c4_remove_citations"],
        min_num_sentences=quality["c4_min_sentences"], min_words_per_line=quality["c4_min_words_per_line"],
        max_word_length=quality["c4_max_word_length"], filter_lorem_ipsum=quality["c4_filter_lorem_ipsum"],
        filter_javascript=quality["c4_filter_javascript"], filter_curly_bracket=quality["c4_filter_curly_bracket"],
        filter_policy=quality["c4_filter_policy"])
    return transformer, directory / "c4_filters.py"


def measure_rows(rows, tokenizer, sentence_writer, document_writer, dataset, condition, cache=None, transformer=None):
    accumulator, measurements, identities = Accumulator(), {}, []
    changed_count = 0
    for raw, document in rows:
        identity = document["document_id"]
        identities.append(identity)
        original_hash = hashlib.sha256(raw).hexdigest()
        if cache is not None and (identity not in cache or cache[identity]["raw_sha256"] != original_hash):
            raise ValueError(f"{dataset}/{condition}: document not identical to its baseline row: {identity}")
        text = document["text"]
        if transformer is not None:
            working = SimpleNamespace(text=text)
            decision = transformer.filter(working)
            accepted = decision[0] if isinstance(decision, tuple) else decision
            if not accepted:
                raise ValueError(f"Saved quality survivor fails reference C4: {identity}")
            text = working.text
            changed_count += text != document["text"]
        lengths = (cache[identity]["lengths"] if cache is not None and transformer is None
                   else tokenizer.sentence_lengths(text))
        accumulator.add(document["domain"], lengths, len(text))
        if cache is None:
            measurements[identity] = {"raw_sha256": original_hash, "lengths": lengths}
        document_writer.writerow({"dataset": dataset, "condition": condition, "document_id": identity,
                                  "domain": document["domain"], "token_count": sum(lengths), "sentence_count": len(lengths),
                                  "character_count": len(text), "source_wet_path": document["source"]["wet_path"],
                                  "source_record_index": document["source"]["record_index"]})
        for index, length in enumerate(lengths):
            sentence_writer.writerow({"dataset": dataset, "condition": condition, "document_id": identity,
                                      "sentence_index": index, "sentence_tokens": length})
    return accumulator, measurements, identities, changed_count


def collect(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(OUTPUT.resolve()):
        raise ValueError("Stage 4 outputs must stay within results/stage4")
    protected_count = verify_preserved()
    if any((directory / name).exists() for name in ("metrics.json", "document_metrics.csv", "sentence_lengths.csv")):
        raise FileExistsError("Metrics output exists. Use a new --output-dir for a separate reproducibility run.")
    directory.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "filter_config.json")
    tokenizer = MeasurementTokenizer(ROOT / config["assets_directory"])
    transformer, c4_source = c4_transformer(tokenizer, config)
    all_metrics, supplemental, effects, domains, histograms, input_records = [], [], [], [], [], []
    with (directory / "sentence_lengths.csv").open("w", encoding="utf-8", newline="") as sentence_stream, \
            (directory / "document_metrics.csv").open("w", encoding="utf-8", newline="") as document_stream:
        sentence_writer = csv.DictWriter(sentence_stream, fieldnames=["dataset", "condition", "document_id", "sentence_index", "sentence_tokens"])
        document_writer = csv.DictWriter(document_stream, fieldnames=["dataset", "condition", "document_id", "domain", "token_count", "sentence_count", "character_count", "source_wet_path", "source_record_index"])
        sentence_writer.writeheader()
        document_writer.writeheader()
        for dataset in (1, 2, 3):
            inputs = condition_inputs(dataset)
            dataset_metrics, cache, baseline, saved_quality = [], None, None, None
            for condition in (*CONDITIONS, "quality_c4_transformed"):
                source_condition = "quality" if condition == "quality_c4_transformed" else condition
                source = inputs[source_condition]
                path = source["path"]
                if sha256_file(path) != source["sha256"]:
                    raise ValueError(f"Input checksum mismatch: {path}")
                accumulator, new_cache, identities, changed = measure_rows(
                    baseline_rows(path, config["max_text_characters"]), tokenizer, sentence_writer, document_writer,
                    dataset, condition, cache=cache, transformer=transformer if condition == "quality_c4_transformed" else None)
                if accumulator.document_count != source["count"]:
                    raise ValueError(f"Wrong input document count: {dataset}/{condition}")
                if condition == "baseline":
                    cache = new_cache
                    baseline = accumulator.summarize()
                    expected_ids = hashlib.sha256("".join(identity + "\n" for identity in identities).encode()).hexdigest()
                    if any(record["input_ids_sha256"] != expected_ids for stage, record in inputs.items() if stage != "baseline"):
                        raise ValueError("A filtering stage used different baseline IDs")
                current = accumulator.summarize()
                row = {"dataset": dataset, "condition": condition, "primary_condition": condition in CONDITIONS,
                       **relative_to_baseline(current, baseline)}
                if condition in CONDITIONS:
                    dataset_metrics.append(row)
                    all_metrics.append(row)
                else:
                    supplemental.append(row)
                    if current["document_count"] != saved_quality["document_count"] or accumulator.domains != quality_domains:
                        raise ValueError("C4 transformation changed retained IDs/domains")
                    effects.append({"dataset": dataset, "retained_documents": current["document_count"], "changed_documents": changed,
                                    "characters_removed_by_c4": saved_quality["character_count"] - current["character_count"],
                                    "saved_quality_tokens": saved_quality["token_count"], "c4_quality_tokens": current["token_count"],
                                    "tokens_removed_by_c4": saved_quality["token_count"] - current["token_count"],
                                    "token_reduction_percentage_of_saved_quality": safe_percentage(saved_quality["token_count"] - current["token_count"], saved_quality["token_count"]),
                                    "tokens_removed_by_document_selection": baseline["token_count"] - saved_quality["token_count"],
                                    "tokens_removed_combined_selection_and_c4": baseline["token_count"] - current["token_count"],
                                    "saved_quality_sentence_count": saved_quality["sentence_count"], "c4_sentence_count": current["sentence_count"],
                                    "sentence_count_change_from_saved_quality": current["sentence_count"] - saved_quality["sentence_count"],
                                    "sentence_mean_change_from_saved_quality": (current["sentence_mean_tokens"] - saved_quality["sentence_mean_tokens"]
                                                                               if current["sentence_mean_tokens"] is not None and saved_quality["sentence_mean_tokens"] is not None else None),
                                    "domain_count_unchanged": True, "domain_entropy_unchanged": current["domain_entropy_bits"] == saved_quality["domain_entropy_bits"]})
                if condition == "quality":
                    saved_quality, quality_domains = current, accumulator.domains.copy()
                domains.extend({"dataset": dataset, "condition": condition, "domain": domain, "documents": count,
                                "proportion": count / accumulator.document_count} for domain, count in sorted(accumulator.domains.items()))
                histograms.extend({"dataset": dataset, "condition": condition, "sentence_tokens": length, "frequency": count}
                                  for length, count in sorted(accumulator.lengths.items()))
                input_records.append({"dataset": dataset, "condition": condition, "input": str(path.relative_to(ROOT)),
                                      "input_sha256": source["sha256"], "documents": accumulator.document_count,
                                      "document_ids_sha256": hashlib.sha256("".join(identity + "\n" for identity in identities).encode()).hexdigest()})
                if sha256_file(path) != source["sha256"]:
                    raise ValueError("Input changed while measuring")
                print(f"Dataset {dataset}/{condition}: {current['document_count']} documents, {current['token_count']} tokens, {current['sentence_count']} sentences", flush=True)
            write_csv(directory / f"dataset_{dataset}_comparison.csv", dataset_metrics)
            write_json(directory / f"dataset_{dataset}_comparison.json", dataset_metrics)
            del cache
    if len(all_metrics) != 15 or len(supplemental) != 3:
        raise ValueError("Wrong number of experimental conditions")
    write_csv(directory / "metrics.csv", all_metrics)
    write_json(directory / "metrics.json", all_metrics)
    write_csv(directory / "quality_c4_metrics.csv", supplemental)
    write_json(directory / "quality_c4_metrics.json", supplemental)
    write_csv(directory / "quality_transform_effects.csv", effects)
    write_json(directory / "quality_transform_effects.json", effects)
    write_csv(directory / "domain_counts.csv", domains)
    write_json(directory / "domain_counts.json", domains)
    write_csv(directory / "sentence_histograms.csv", histograms)
    write_json(directory / "sentence_histograms.json", histograms)
    manifest = {"schema_version": 1, "primary_combinations": 15, "supplementary_combinations": 3,
                "inputs": input_records, "protected_files_unchanged": verify_preserved(),
                "measurement": {"tokens": "English Punkt sentences then NLTKWordTokenizer; punctuation counted; no normalization",
                                "tokenizer_asset": tokenizer.tokenizer.asset_record,
                                "standard_deviation": "population, ddof=0", "percentiles": "Hyndman-Fan type 7 linear interpolation",
                                "domain": "exact existing hostname field, one count per document", "entropy_log_base": 2,
                                "zero_denominator_percentage": None, "empty_sentence_statistics": None,
                                "c4_reference_sha256": sha256_file(c4_source)},
                "versions": {"python": platform.python_version(), "nltk": importlib.metadata.version("nltk")},
                "implementation_sha256": {path.name: sha256_file(path) for path in sorted((ROOT / "stage4").glob("*.py"))},
                "filter_config_sha256": sha256_file(ROOT / "filter_config.json")}
    assert manifest["protected_files_unchanged"] == protected_count
    manifest["output_sha256"] = {path.name: sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}
    write_json(directory / "manifest.json", manifest)
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    collect(arguments.output_dir)
