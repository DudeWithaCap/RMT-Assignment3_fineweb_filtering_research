"""Resumable large-sample acquisition: the Dataset 1 procedure at scale.

The sampling method is identical to src.prepare_dataset: WET paths are ordered by
SHA256(seed:path) and consecutive eligible conversion records are taken from each file
until the target count is reached. Documents are built by the same
document_from_record(), so the first documents equal the original Dataset 1 sample.

One deliberate difference: a record that fails the project's document validation (for
example a URL whose parsed host differs from the stored domain) is skipped and counted
with example URLs, where the original script would stop with an error.

Unlike the original, each source file is cached as a verified part under
data/raw/<name>/parts/, so a temporary Common Crawl outage (HTTP 503) resumes from the
next missing file instead of restarting a multi-gigabyte download.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

from src.filter_assets import ROOT, sha256_file
from src.prepare_dataset import BASE_URL, document_from_record, load_config, ordered_paths

RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class CountingReader:
    def __init__(self, raw, budget):
        self.raw, self.budget = raw, budget

    def read(self, size=-1):
        if self.budget["used"] >= self.budget["limit"]:
            raise RuntimeError("Compressed download budget exhausted")
        chunk = self.raw.read(65536 if size is None or size < 0 else size)
        self.budget["used"] += len(chunk)
        return chunk


def session_factory():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update({"User-Agent": "RMT-Research-Acquisition/1.0", "Accept-Encoding": "identity"})
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=6, backoff_factor=4, status_forcelist=RETRYABLE_STATUS, respect_retry_after_header=True)))
    return session


def with_retries(action, label, attempts, pause, log=print, sleep=time.sleep):
    """Retry a whole-file action; per-request retries happen inside the session."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except KeyboardInterrupt:
            raise
        except Exception as error:
            if attempt == attempts:
                raise
            wait = min(pause * 2 ** (attempt - 1), 900)
            log(f"{label}: attempt {attempt} failed ({type(error).__name__}: {error}); retrying in {wait}s")
            sleep(wait)


def write_json(path, value):
    partial = Path(str(path) + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


def fetch_manifest(session, crawl, work, timeout, budget):
    cached = work / "wet.paths"
    if not cached.exists():
        with session.get(BASE_URL + f"crawl-data/{crawl}/wet.paths.gz", stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=CountingReader(response.raw, budget)) as stream:
                manifest = stream.read(32 * 1024 * 1024 + 1)
        if len(manifest) > 32 * 1024 * 1024:
            raise RuntimeError("Unexpectedly large WET manifest")
        Path(str(cached) + ".partial").write_bytes(manifest)
        Path(str(cached) + ".partial").replace(cached)
    manifest = cached.read_bytes()
    paths = manifest.decode("utf-8").splitlines()
    prefix = f"crawl-data/{crawl}/segments/"
    if not paths or any(not p.startswith(prefix) or not p.endswith(".warc.wet.gz") or ".." in p for p in paths):
        raise ValueError("Invalid official WET manifest")
    return paths, hashlib.sha256(manifest).hexdigest()


def fetch_part(session, path, destination, limit, timeout, budget):
    """Stream one WET file into a part of eligible documents, published atomically."""
    from warcio.archiveiterator import ArchiveIterator

    partial = Path(str(destination) + ".partial")
    skipped, digest, documents, records = Counter(), hashlib.sha256(), 0, 0
    invalid_examples = []
    with session.get(BASE_URL + path, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as target:
            for index, record in enumerate(ArchiveIterator(CountingReader(response.raw, budget))):
                records += 1
                try:
                    document, reason = document_from_record(record, path, index, limit)
                except ValueError as error:
                    # The original Dataset 1 code would abort here; a record failing the
                    # project's own document validation is excluded and counted instead.
                    document, reason = None, f"invalid_document: {error}"
                    if len(invalid_examples) < 20:
                        invalid_examples.append({"record_index": index, "error": str(error),
                                                 "url": record.rec_headers.get_header("WARC-Target-URI")})
                if reason:
                    skipped[reason] += 1
                    continue
                line = (json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                target.write(line)
                digest.update(line)
                documents += 1
    if not documents:
        raise RuntimeError(f"No eligible documents in {path}")
    partial.replace(destination)
    meta = {"wet_path": path, "documents": documents, "records_seen": records,
            "skipped_records": dict(skipped), "invalid_examples": invalid_examples, "sha256": digest.hexdigest(),
            "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(destination.with_suffix(".json"), meta)
    return meta


def verified_part(destination):
    meta_path = destination.with_suffix(".json")
    if not destination.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta if sha256_file(destination) == meta["sha256"] else None


def assemble(parts, output, target_count):
    """Concatenate parts in source order, stopping exactly at the target count."""
    partial = Path(str(output) + ".partial")
    digest, domains, written = hashlib.sha256(), Counter(), 0
    stats = {"total_characters": 0, "original_total_characters": 0, "truncated_documents": 0,
             "empty_documents": 0, "min_characters": None, "max_characters": None}
    with partial.open("wb") as target:
        for part in parts:
            with part.open("rb") as stream:
                for line in stream:
                    if written == target_count:
                        break
                    document = json.loads(line)
                    target.write(line)
                    digest.update(line)
                    written += 1
                    length = document["character_count"]
                    domains[document["domain"]] += 1
                    stats["total_characters"] += length
                    stats["original_total_characters"] += document["original_character_count"]
                    stats["truncated_documents"] += document["truncated"]
                    stats["empty_documents"] += length == 0
                    stats["min_characters"] = length if stats["min_characters"] is None else min(stats["min_characters"], length)
                    stats["max_characters"] = length if stats["max_characters"] is None else max(stats["max_characters"], length)
            if written == target_count:
                break
    if written != target_count:
        partial.unlink()
        raise RuntimeError(f"Only {written} documents available in downloaded parts")
    partial.replace(output)
    stats.update(document_count=written, unique_domains=len(domains),
                 mean_characters=stats["total_characters"] / written,
                 top_domains=dict(domains.most_common(100)))
    return digest.hexdigest(), stats


def prepare_large(config, output_name, root=ROOT, session=None, log=print, sleep=time.sleep,
                  file_attempts=8, retry_pause=30):
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.jsonl", output_name) or output_name == "baseline.jsonl":
        raise ValueError("Choose a new plain .jsonl name; the original baseline.jsonl is never replaced")
    processed = root / "data/processed"
    output = processed / output_name
    work = root / "data/raw" / Path(output_name).stem
    metadata_path = root / "results" / f"{Path(output_name).stem}_metadata.json"
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"{output_name} already exists; move it before re-acquiring")
    for directory in (processed, work / "parts", metadata_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    budget = {"limit": config["max_download_bytes"], "used": 0}
    timeout = (15, config["request_timeout_seconds"])
    started = datetime.now(timezone.utc).isoformat()
    session = session or session_factory()
    paths, manifest_hash = with_retries(
        lambda: fetch_manifest(session, config["crawl"], work, timeout, budget), "manifest", file_attempts, retry_pause, log, sleep)
    selected = ordered_paths(paths, config["sampling_seed"])[:config["max_source_files"]]
    write_json(work / "source_selection.json", {"crawl": config["crawl"], "seed": config["sampling_seed"],
               "manifest_sha256": manifest_hash, "selected_paths": selected})
    parts, part_records, available = [], [], 0
    for number, path in enumerate(selected):
        destination = work / "parts" / f"{number:04d}.jsonl"
        meta = verified_part(destination)
        if meta is None or meta["wet_path"] != path:
            log(f"[{available:,}/{config['dataset_size']:,}] downloading file {number + 1}: {path}")
            meta = with_retries(lambda: fetch_part(session, path, destination, config["max_text_characters"],
                                                   timeout, budget),
                                f"file {number + 1}", file_attempts, retry_pause, log, sleep)
        else:
            log(f"[{available:,}/{config['dataset_size']:,}] reusing cached file {number + 1}")
        parts.append(destination)
        part_records.append(meta)
        available += meta["documents"]
        if available >= config["dataset_size"]:
            break
    if available < config["dataset_size"]:
        raise RuntimeError(f"Only {available} documents within max_source_files={config['max_source_files']}")
    log("Assembling baseline...")
    checksum, statistics = assemble(parts, output, config["dataset_size"])
    metadata = {
        "status": "complete", "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "output": str(output.relative_to(root)), "dataset_sha256": checksum,
        "dataset_bytes": output.stat().st_size, "source": "Official Common Crawl WET (extracted plain text)",
        "manifest_url": BASE_URL + f"crawl-data/{config['crawl']}/wet.paths.gz", "manifest_sha256": manifest_hash,
        "sampling_method": "Same as Dataset 1: sort WET paths by SHA256(seed:path); take consecutive conversion "
                           "records from each selected file until target; no random document sampling",
        "files": part_records, "compressed_bytes_read_this_session": budget["used"], "statistics": statistics,
        "limitations": [
            "Prefix/cluster sample of whole WET files, not representative of Common Crawl or the web.",
            "Common Crawl has already extracted text; HTML extraction is not evaluated.",
            "UTF-8 decoding replaces invalid bytes; text truncated to configured characters.",
            "Domain means lowercase hostname, including subdomains."],
        "python_version": platform.python_version(),
        "library_versions": {name: importlib.metadata.version(name) for name in ("requests", "warcio", "urllib3")},
    }
    write_json(metadata_path, metadata)
    log(f"Saved {config['dataset_size']:,} documents to {output}")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config_1m.json")
    parser.add_argument("--output-name", default="baseline_1m.jsonl", help="File name inside data/processed/")
    args = parser.parse_args(argv)
    try:
        prepare_large(load_config(args.config), args.output_name, log=lambda m: print(m, flush=True))
    except (Exception, KeyboardInterrupt) as error:
        print(f"Large acquisition stopped: {error}. Rerun the same command to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
