"""Run independent filters against one immutable Common Crawl baseline."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time

from .filter_assets import DATATROVE_REVISION, ROOT, sha256_file
from .prepare_dataset import validate_document

STAGES = ("url", "language", "quality", "dedup")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))

    def number(value, label, low=0, high=None, integer=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < low or (high is not None and value > high) or (integer and type(value) is not int)):
            raise ValueError(f"Invalid configuration value: {label}")

    for key in ("baseline", "assets_directory"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"Invalid path: {key}")
    checksum = config["baseline_sha256"]
    if checksum is not None and (not isinstance(checksum, str) or len(checksum) != 64
                                 or any(c not in "0123456789abcdef" for c in checksum)):
        raise ValueError("Invalid baseline SHA256")
    number(config["expected_documents"], "expected_documents", low=1, integer=True)
    number(config["max_text_characters"], "max_text_characters", low=1, high=20000, integer=True)
    number(config["url"]["soft_word_threshold"], "soft_word_threshold", low=1, integer=True)
    for key in ("extra_domains", "extra_urls"):
        if not isinstance(config["url"][key], list) or any(not isinstance(v, str) or not v for v in config["url"][key]):
            raise ValueError(f"Invalid URL {key}")
    number(config["language"]["english_confidence_threshold"], "english_confidence_threshold", high=1)
    if config["language"]["model_file"] != "lid.176.ftz":
        raise ValueError("This acquisition setup explicitly uses lid.176.ftz; other models require documented asset setup")
    number(config["dedup"]["ngram_size"], "ngram_size", low=1, high=20, integer=True)
    number(config["dedup"]["similarity_threshold"], "similarity_threshold", low=0.000001, high=1)
    number(config["dedup"]["max_documents"], "max_documents", low=1, integer=True)
    quality = config["quality"]
    integer_keys = ("min_doc_words", "max_doc_words", "min_stop_words", "c4_min_sentences",
                    "c4_min_words_per_line", "c4_max_word_length", "short_line_length")
    ratio_keys = ("max_symbol_word_ratio", "max_bullet_lines_ratio", "max_ellipsis_lines_ratio", "min_alpha_word_ratio",
                  "dup_line_frac", "dup_para_frac", "dup_line_char_frac", "dup_para_char_frac", "line_punct_threshold",
                  "short_line_threshold", "char_duplicates_ratio", "newline_word_ratio")
    for key in integer_keys:
        number(quality[key], key, low=1 if key in ("max_doc_words", "c4_max_word_length") else 0, integer=True)
    for key in ratio_keys:
        number(quality[key], key, high=1)
    for key in ("min_avg_word_length", "max_avg_word_length"):
        number(quality[key], key)
    if quality["min_doc_words"] > quality["max_doc_words"] or quality["min_avg_word_length"] > quality["max_avg_word_length"]:
        raise ValueError("Minimum quality bounds must not exceed maximum bounds")
    for key in ("top_ngram_thresholds", "duplicate_ngram_thresholds"):
        if not isinstance(quality[key], dict):
            raise ValueError(f"Invalid n-gram threshold map: {key}")
        for n, threshold in quality[key].items():
            number(int(n), key, low=1, high=20, integer=True)
            number(threshold, key, high=1)
    for key in ("c4_remove_citations", "c4_filter_lorem_ipsum", "c4_filter_javascript",
                "c4_filter_curly_bracket", "c4_filter_policy"):
        if type(quality[key]) is not bool:
            raise ValueError(f"Invalid boolean: {key}")
    if not isinstance(quality["stop_words"], list) or any(not isinstance(v, str) for v in quality["stop_words"]):
        raise ValueError("Invalid stop word list")
    return config


def baseline_rows(path, limit):
    seen = set()
    with Path(path).open("rb") as stream:
        for index, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Blank baseline row {index}")
            document = json.loads(line)
            validate_document(document, limit)
            if document["document_id"] in seen:
                raise ValueError(f"Repeated document ID at row {index}")
            seen.add(document["document_id"])
            yield line, document


def construct_filter(stage, config, assets, workspace):
    if stage == "url":
        from .url_filter import URLFilter
        return URLFilter(config[stage], assets, workspace)
    if stage == "language":
        from .language_filter import LanguageFilter
        return LanguageFilter(config[stage], assets)
    from .filter_tokenizer import EnglishTokenizer
    tokenizer = EnglishTokenizer(assets)
    if stage == "quality":
        from .quality_filter import QualityFilter
        return QualityFilter(config[stage], tokenizer)
    from .dedup_filter import DedupFilter
    return DedupFilter(config[stage], tokenizer, workspace)


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_stage(stage, config, root=ROOT, overwrite=False, factory=construct_filter):
    if stage not in STAGES:
        raise ValueError(f"Unknown filter stage: {stage}")
    root = Path(root)
    baseline = (root / config["baseline"]).resolve()
    processed, results = root / "data/processed", root / "results"
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dataset": processed / f"filtered_{stage}.jsonl",
        "decisions": results / f"{stage}_decisions.jsonl",
        "removed": results / f"{stage}_removed.jsonl",
        "summary": results / f"{stage}_summary.json",
    }
    if any(path.resolve() == baseline for path in outputs.values()):
        raise ValueError("Filter output may never overwrite the baseline")
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"{stage} outputs exist; use --overwrite to replace filter outputs only")
    started = time.perf_counter()
    report = {"stage": stage, "status": "failed", "baseline": str(baseline),
              "started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
              "datatrove_revision": DATATROVE_REVISION, "input_count": 0, "output_count": 0}
    evaluator = None
    try:
        before = sha256_file(baseline)
        report["baseline_sha256"] = before
        if config["baseline_sha256"] is not None and before != config["baseline_sha256"]:
            raise ValueError("Baseline does not match the configured immutable baseline checksum")
        input_count = sum(1 for _ in baseline_rows(baseline, config["max_text_characters"]))
        report["input_count"] = input_count
        if input_count != config["expected_documents"]:
            raise ValueError(f"Expected {config['expected_documents']} baseline documents, found {input_count}")
        assets = root / config["assets_directory"]
        with tempfile.TemporaryDirectory(prefix=f"{stage}_work_", dir=results) as directory:
            workspace = Path(directory)
            try:
                evaluator = factory(stage, config, assets, workspace)
                decision_map = None
                if stage == "dedup":
                    for _, document in baseline_rows(baseline, config["max_text_characters"]):
                        evaluator.add(document)
                    decision_map = evaluator.decisions()
                digest, ids_digest, output_digest, decision_digest = (hashlib.sha256() for _ in range(4))
                count, kept, reasons = 0, 0, Counter()
                with (workspace / "dataset").open("wb") as target, (workspace / "decisions").open("wb") as decisions, \
                        (workspace / "removed").open("wb") as removed:
                    for line, document in baseline_rows(baseline, config["max_text_characters"]):
                        digest.update(line)
                        ids_digest.update((document["document_id"] + "\n").encode())
                        count += 1
                        accepted, reason, details = (decision_map[document["document_id"]]
                                                     if decision_map is not None else evaluator.evaluate(document))
                        decision = {"document_id": document["document_id"], "url": document["url"],
                                    "domain": document["domain"], "source": document["source"],
                                    "stage": stage, "kept": accepted, "reason": reason, "details": details}
                        decision_line = (json.dumps(decision, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
                        decisions.write(decision_line)
                        decision_digest.update(decision_line)
                        if accepted:
                            target.write(line)
                            output_digest.update(line)
                            kept += 1
                        else:
                            removed.write(decision_line)
                            reasons[reason] += 1
                after = sha256_file(baseline)
                report["baseline_sha256_after"] = after
                if after != before or digest.hexdigest() != before or count != input_count:
                    raise RuntimeError("Baseline changed during filtering; outputs not published")
                report.update({"status": "complete", "output_count": kept, "removed_count": count - kept,
                               "removal_reasons": dict(sorted(reasons.items())),
                               "input_ids_sha256": ids_digest.hexdigest(), "output_sha256": output_digest.hexdigest(),
                               "decisions_sha256": decision_digest.hexdigest(), "assets": evaluator.assets,
                               "python_version": platform.python_version(),
                               "dependencies": {name: importlib.metadata.version(name) for name in
                                                ("fasttext-community", "numpy", "nltk", "tldextract")},
                               "implementation_sha256": {p.name: sha256_file(p) for p in sorted((ROOT / "src").glob("*.py"))},
                               "elapsed_seconds": time.perf_counter() - started,
                               "completed_utc": datetime.now(timezone.utc).isoformat()})
                write_json(workspace / "summary", report)
                # Each artifact is replaced atomically; summary is published last.
                for key, destination in outputs.items():
                    (workspace / key).replace(destination)
                failure = results / f"{stage}_failure.json"
                if failure.exists():
                    failure.replace(results / f"{stage}_previous_failure.json")
            finally:
                if evaluator is not None and hasattr(evaluator, "close"):
                    evaluator.close()
        print(f"{stage}: {input_count} input -> {kept} kept ({input_count - kept} removed)", flush=True)
        return report
    except (Exception, KeyboardInterrupt) as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                       "elapsed_seconds": time.perf_counter() - started})
        write_json(results / f"{stage}_failure.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "filter_config.json")
    parser.add_argument("--overwrite", action="store_true", help="Replace filter outputs only; baseline remains immutable")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except Exception as error:
        print(f"Invalid filter configuration: {error}", file=sys.stderr)
        return 1
    failed = False
    for stage in STAGES if args.stage == "all" else (args.stage,):
        try:
            run_stage(stage, config, overwrite=args.overwrite)
        except (Exception, KeyboardInterrupt) as error:
            print(f"{stage} failed: {error}", file=sys.stderr)
            failed = True
            if isinstance(error, KeyboardInterrupt):
                break
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
