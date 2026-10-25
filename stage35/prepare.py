"""Stream bounded real WET windows and select deterministic hash samples."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import json
from pathlib import Path

import requests
from warcio.archiveiterator import ArchiveIterator

from src.filter_assets import ROOT, sha256_file
from src.prepare_dataset import BASE_URL, BudgetReader, document_from_record, ordered_paths
from .common import EVIDENCE, baseline_path, check_legacy, ledger_rows, write_json


def sampling_config(path=ROOT / "stage35_config.json"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("dataset_size", "files_per_dataset", "candidate_records_per_file", "max_text_characters",
                "max_download_bytes", "max_candidate_cache_bytes", "request_timeout_seconds"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Invalid positive integer: {key}")
    if config["dataset_size"] != 1000 or config["files_per_dataset"] != 2 or config["max_text_characters"] > 20000:
        raise ValueError("Stage 3.5 requires 1,000 documents across two source windows and a maximum 20,000-character cap")
    if config["candidate_records_per_file"] < config["dataset_size"] // config["files_per_dataset"]:
        raise ValueError("Candidate window is smaller than the sample")
    if type(config["sampling_seed"]) is not int:
        raise ValueError("sampling_seed must be an integer")
    return config


def priority(document_id, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}:{document_id}".encode()).digest(), "big")


def select_window(path, count, seed):
    """Lowest hash scores without holding the whole candidate window in RAM."""
    selected = []
    candidates = 0
    with Path(path).open("rb") as stream:
        for ordinal, line in enumerate(stream):
            row = json.loads(line)
            score = priority(row["document_id"], seed)
            item = (-score, -ordinal, line)
            if len(selected) < count:
                heapq.heappush(selected, item)
            elif item > selected[0]:
                heapq.heapreplace(selected, item)
            candidates += 1
    if candidates < count:
        raise RuntimeError(f"Only {candidates} eligible candidates, need {count}")
    # Preserve source-record order after selecting by hash; never sort by language.
    return [item[2] for item in sorted(selected, key=lambda item: -item[1])], candidates


def distinct_segment_paths(paths, seed, count, excluded=()):
    selected, segments = [], set()
    for path in ordered_paths(paths, seed):
        segment = path.split("/")[3]
        if path not in excluded and segment not in segments:
            selected.append(path)
            segments.add(segment)
            if len(selected) == count:
                return selected
    raise ValueError("Not enough distinct source segments")


def acquire(dataset, config):
    check_legacy()
    output = baseline_path(dataset)
    evidence = EVIDENCE / f"dataset_{dataset}"
    metadata_path = evidence / "preparation.json"
    if output.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(output) != metadata["dataset_sha256"] or metadata["sampling_config"] != config:
            raise ValueError("Existing new baseline/config mismatch; nothing overwritten")
        print(f"Dataset {dataset}: verified existing baseline", flush=True)
        return metadata
    partial = output.with_suffix(".jsonl.partial")
    if partial.exists():
        raise FileExistsError(f"Archive incomplete output before retry: {partial}")
    crawl = config["datasets"][str(dataset)]["crawl"]
    raw = ROOT / f"data/raw/dataset_{dataset}"
    raw.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    budget = {"used": 0, "limit": config["max_download_bytes"]}
    report = {"dataset": int(dataset), "crawl": crawl, "status": "in_progress", "sampling_config": config,
              "started_utc": datetime.now(timezone.utc).isoformat(), "frames": [],
              "sampling_method": "Two hash-ranked WET files from distinct segments; in each, select 500 lowest SHA256(seed:document_id) scores within the first 2500 eligible records (or EOF); keep source order",
              "language_or_content_selection": False}
    try:
        timeout = (15, config["request_timeout_seconds"])
        with requests.Session() as session:
            session.headers.update({"User-Agent": "RMT-Stage35-Research/1.0", "Accept-Encoding": "identity"})
            manifest_url = BASE_URL + f"crawl-data/{crawl}/wet.paths.gz"
            with session.get(manifest_url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with gzip.GzipFile(fileobj=BudgetReader(response.raw, budget)) as stream:
                    manifest = stream.read(32 * 1024 * 1024 + 1)
                if len(manifest) > 32 * 1024 * 1024:
                    raise ValueError("Manifest exceeds decompressed limit")
            paths = manifest.decode().splitlines()
            prefix = f"crawl-data/{crawl}/segments/"
            if not paths or any(not path.startswith(prefix) or not path.endswith(".warc.wet.gz") or ".." in path for path in paths):
                raise ValueError("Invalid official Common Crawl WET manifest")
            existing_paths = {row["source"]["wet_path"] for row in ledger_rows(baseline_path(1))}
            other = "3" if str(dataset) == "2" else "2"
            if baseline_path(other).exists():
                existing_paths.update(row["source"]["wet_path"] for row in ledger_rows(baseline_path(other)))
            selected = distinct_segment_paths(paths, config["sampling_seed"], config["files_per_dataset"], existing_paths)
            (raw / "wet.paths").write_bytes(manifest)
            report.update({"manifest_url": manifest_url, "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                           "selected_wet_paths": selected})
            cache_bytes = 0
            digest = hashlib.sha256()
            per_file = config["dataset_size"] // len(selected)
            with partial.open("wb") as target:
                for file_index, source_path in enumerate(selected, 1):
                    cache = raw / f"candidates_{file_index}.jsonl"
                    if cache.exists() or cache.with_suffix(".jsonl.partial").exists():
                        raise FileExistsError(f"Acquisition cache exists without a completed baseline: {cache}; archive it before retrying")
                    frame_partial = cache.with_suffix(".jsonl.partial")
                    eligible, seen, skipped = 0, 0, Counter()
                    print(f"Dataset {dataset}: streaming candidate window {file_index}/2: {source_path}", flush=True)
                    with session.get(BASE_URL + source_path, stream=True, timeout=timeout) as response:
                        response.raise_for_status()
                        with frame_partial.open("wb") as frame:
                            for index, record in enumerate(ArchiveIterator(BudgetReader(response.raw, budget))):
                                seen += 1
                                document, reason = document_from_record(record, source_path, index, config["max_text_characters"])
                                if reason:
                                    skipped[reason] += 1
                                    continue
                                line = (json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                                cache_bytes += len(line)
                                if cache_bytes > config["max_candidate_cache_bytes"]:
                                    raise RuntimeError("Candidate cache budget exhausted")
                                frame.write(line)
                                eligible += 1
                                if eligible == config["candidate_records_per_file"]:
                                    break
                    frame_partial.rename(cache)
                    sample, candidates = select_window(cache, per_file, config["sampling_seed"])
                    for line in sample:
                        target.write(line)
                        digest.update(line)
                    report["frames"].append({"wet_path": source_path, "source_url": BASE_URL + source_path,
                                              "records_seen": seen, "eligible_candidates": candidates,
                                              "selected_count": len(sample), "skipped_records": dict(skipped),
                                              "candidate_cache": str(cache.relative_to(ROOT)),
                                              "candidate_cache_sha256": sha256_file(cache),
                                              "selected_record_indices": [json.loads(line)["source"]["record_index"] for line in sample]})
                    print(f"Dataset {dataset}: selected {len(sample)} of {candidates} eligible candidates", flush=True)
            identities, hostnames, row_count, truncated_count, total_characters = set(), set(), 0, 0, 0
            for row in ledger_rows(partial):
                identities.add(row["document_id"])
                hostnames.add(row["domain"])
                row_count += 1
                truncated_count += row["truncated"]
                total_characters += row["character_count"]
            if row_count != config["dataset_size"] or len(identities) != row_count:
                raise ValueError("New baseline has incorrect size or repeated document IDs")
            report.update({"status": "complete", "dataset_sha256": digest.hexdigest(), "document_count": row_count,
                           "truncated_documents": truncated_count,
                           "unique_hostnames": len(hostnames),
                           "total_characters": total_characters,
                           "compressed_bytes_read": budget["used"], "candidate_cache_bytes": cache_bytes,
                           "completed_utc": datetime.now(timezone.utc).isoformat()})
            partial.rename(output)
            write_json(metadata_path, report)
        check_legacy()
        return report
    except (Exception, KeyboardInterrupt) as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "compressed_bytes_read": budget["used"]})
        write_json(evidence / "preparation_failure.json", report)
        raise


def replay_selection(dataset, config):
    metadata = json.loads((EVIDENCE / f"dataset_{dataset}/preparation.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for frame in metadata["frames"]:
        cache = ROOT / frame["candidate_cache"]
        if sha256_file(cache) != frame["candidate_cache_sha256"]:
            raise ValueError("Candidate cache changed")
        selected, _ = select_window(cache, frame["selected_count"], config["sampling_seed"])
        for line in selected:
            digest.update(line)
    assert digest.hexdigest() == metadata["dataset_sha256"] == sha256_file(baseline_path(dataset))
    return {"dataset": int(dataset), "selection_replayed_from_cached_real_windows": True, "sha256": digest.hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["2", "3", "all"], default="all")
    args = parser.parse_args()
    settings = sampling_config()
    for dataset in ("2", "3") if args.dataset == "all" else (args.dataset,):
        acquire(dataset, settings)
        write_json(EVIDENCE / f"dataset_{dataset}/sampling_replay.json", replay_selection(dataset, settings))
