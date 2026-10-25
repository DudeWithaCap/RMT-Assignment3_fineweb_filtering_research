"""Stream a deterministic prefix sample from official Common Crawl WET files."""

import argparse
import codecs
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import re
import sys
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from warcio.archiveiterator import ArchiveIterator

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://data.commoncrawl.org/"


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not re.fullmatch(r"CC-MAIN-\d{4}-\d{2}", config.get("crawl", "")):
        raise ValueError("crawl must be a pinned CC-MAIN-YYYY-WW identifier")
    for key in ("dataset_size", "max_text_characters", "max_source_files",
                "max_download_bytes", "request_timeout_seconds"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_text_characters"] > 20000:
        raise ValueError("Stage 1 limits text to at most 20,000 characters")
    if type(config.get("sampling_seed")) is not int:
        raise ValueError("sampling_seed must be an integer")
    return config


class BudgetReader:
    """Bound compressed response reads, counting bytes across all requests."""

    def __init__(self, raw, budget):
        self.raw = raw
        self.budget = budget

    def read(self, size=-1):
        remaining = self.budget["limit"] - self.budget["used"]
        if remaining <= 0:
            raise RuntimeError("Compressed download budget exhausted; no dataset published")
        size = min(65536 if size < 0 else size, remaining)
        chunk = self.raw.read(size)
        self.budget["used"] += len(chunk)
        return chunk


def ordered_paths(paths, seed):
    # Hash ranking avoids dependence on Python's random implementation.
    return sorted(paths, key=lambda path: (
        hashlib.sha256(f"{seed}:{path}".encode("utf-8")).digest(), path))


def read_text(stream, limit):
    """Decode incrementally, keeping only a prefix and counting the full text."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    parts, kept, total = [], 0, 0
    while True:
        chunk = stream.read(65536)
        text = decoder.decode(chunk, final=not chunk)
        total += len(text)
        if kept < limit:
            prefix = text[:limit - kept]
            parts.append(prefix)
            kept += len(prefix)
        if not chunk:
            break
    return "".join(parts), total


def document_from_record(record, source_path, index, limit):
    if record.rec_type != "conversion":
        return None, "non_conversion"
    url = record.rec_headers.get_header("WARC-Target-URI")
    try:
        parsed = urlsplit(url or "")
        domain = parsed.hostname
        valid = parsed.scheme.lower() in {"http", "https"} and domain
    except ValueError:
        valid = False
    if not valid:
        return None, "missing_or_invalid_url"
    record_id = record.rec_headers.get_header("WARC-Record-ID")
    if not record_id:
        return None, "missing_record_id"
    text, original_count = read_text(record.content_stream(), limit)
    # Keep empty documents, duplicates, all languages, boilerplate and short text.
    document = {
        "document_id": hashlib.sha256(f"{source_path}:{index}:{record_id}".encode()).hexdigest(),
        "text": text,
        "url": url,
        "domain": domain.lower(),
        "character_count": len(text),
        "original_character_count": original_count,
        "truncated": original_count > len(text),
        "source": {
            "wet_path": source_path,
            "record_index": index,
            "warc_record_id": record_id,
            "warc_date": record.rec_headers.get_header("WARC-Date"),
        },
    }
    validate_document(document, limit)
    return document, None


def validate_document(document, limit=20000):
    for key in ("document_id", "text", "url", "domain"):
        if not isinstance(document.get(key), str):
            raise ValueError(f"Invalid {key}")
    if not document["document_id"] or not document["domain"]:
        raise ValueError("Missing document identity or domain")
    if urlsplit(document["url"]).hostname != document["domain"]:
        raise ValueError("Domain does not match source URL")
    length = len(document["text"])
    if length > limit or document["character_count"] != length:
        raise ValueError("Invalid text length")
    if document["original_character_count"] < length:
        raise ValueError("Invalid original text length")
    if document["truncated"] != (document["original_character_count"] > length):
        raise ValueError("Invalid truncation flag")
    source = document.get("source", {})
    if not source.get("wet_path") or not source.get("warc_record_id"):
        raise ValueError("Missing Common Crawl provenance")
    if type(source.get("record_index")) is not int or source["record_index"] < 0:
        raise ValueError("Invalid source record index")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(config, root=ROOT):
    raw, processed, results = (root / p for p in ("data/raw", "data/processed", "results"))
    for directory in (raw, processed, results):
        directory.mkdir(parents=True, exist_ok=True)
    output = processed / "baseline.jsonl"
    partial = processed / "baseline.jsonl.partial"
    metadata_path = results / "dataset_metadata.json"
    if any(p.exists() for p in (output, partial, metadata_path)):
        raise FileExistsError("Baseline outputs already exist. Archive/move them before rerunning.")
    budget = {"limit": config["max_download_bytes"], "used": 0}
    metadata = {
        "status": "in_progress",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "source": "Official Common Crawl WET (extracted plain text)",
        "manifest_url": BASE_URL + f"crawl-data/{config['crawl']}/wet.paths.gz",
        "sampling_method": "Sort WET paths by SHA256(seed:path); take consecutive conversion records from each selected file until target; no random document sampling",
        "limitations": [
            "Prefix/cluster sample, not representative of Common Crawl or the web.",
            "Common Crawl has already extracted text; cannot evaluate HTML extraction with this baseline.",
            "UTF-8 decoding replaces invalid bytes; text truncated to configured characters.",
            "Domain means lowercase hostname, including subdomains; not registrable domain.",
        ],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": {name: importlib.metadata.version(name) for name in
                             ("requests", "warcio", "urllib3", "six", "certifi", "charset-normalizer", "idna")},
        "files_accessed": [],
    }
    skipped, domains, lengths = Counter(), Counter(), []
    digest = hashlib.sha256()
    truncated, original_characters, records_seen = 0, 0, 0
    try:
        with requests.Session() as session:
            session.headers.update({"User-Agent": "RMT-Stage1-Research/1.0", "Accept-Encoding": "identity"})
            session.mount("https://", HTTPAdapter(max_retries=Retry(
                total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
            timeout = (15, config["request_timeout_seconds"])
            with session.get(metadata["manifest_url"], stream=True, timeout=timeout) as response:
                response.raise_for_status()
                compressed = BudgetReader(response.raw, budget)
                with gzip.GzipFile(fileobj=compressed) as stream:
                    manifest = stream.read(32 * 1024 * 1024 + 1)
                if len(manifest) > 32 * 1024 * 1024:
                    raise RuntimeError("Unexpectedly large WET manifest")
            paths = manifest.decode("utf-8").splitlines()
            prefix = f"crawl-data/{config['crawl']}/segments/"
            if not paths or any(not p.startswith(prefix) or not p.endswith(".warc.wet.gz") or ".." in p for p in paths):
                raise ValueError("Invalid official WET manifest")
            metadata["manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
            selected = ordered_paths(paths, config["sampling_seed"])[:config["max_source_files"]]
            (raw / "wet.paths").write_bytes(manifest)
            write_json(raw / "source_selection.json", {"crawl": config["crawl"],
                       "seed": config["sampling_seed"], "selected_paths": selected,
                       "manifest_sha256": metadata["manifest_sha256"]})
            with partial.open("wb") as target:
                for path in selected:
                    print(f"Streaming {path}", flush=True)
                    metadata["files_accessed"].append(path)
                    with session.get(BASE_URL + path, stream=True, timeout=timeout) as response:
                        response.raise_for_status()
                        for index, record in enumerate(ArchiveIterator(BudgetReader(response.raw, budget))):
                            records_seen += 1
                            document, reason = document_from_record(record, path, index, config["max_text_characters"])
                            if reason:
                                skipped[reason] += 1
                                continue
                            line = (json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                            target.write(line)
                            digest.update(line)
                            lengths.append(document["character_count"])
                            domains[document["domain"]] += 1
                            truncated += document["truncated"]
                            original_characters += document["original_character_count"]
                            if len(lengths) % 100 == 0:
                                print(f"Prepared {len(lengths)}/{config['dataset_size']}", flush=True)
                            if len(lengths) == config["dataset_size"]:
                                break
                    if len(lengths) == config["dataset_size"]:
                        break
            if len(lengths) != config["dataset_size"]:
                raise RuntimeError(f"Only {len(lengths)} documents found within max_source_files")
        metadata.update({
            "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": digest.hexdigest(), "dataset_bytes": partial.stat().st_size,
            "compressed_bytes_read": budget["used"], "records_seen": records_seen,
            "skipped_records": dict(skipped),
            "statistics": {"document_count": len(lengths), "unique_domains": len(domains),
                           "total_characters": sum(lengths), "original_total_characters": original_characters,
                           "min_characters": min(lengths), "max_characters": max(lengths),
                           "mean_characters": sum(lengths) / len(lengths),
                           "truncated_documents": truncated, "empty_documents": lengths.count(0),
                           "domain_counts": dict(sorted(domains.items()))},
        })
        partial.rename(output)
        write_json(metadata_path, metadata)
        print(f"Saved {len(lengths)} documents to {output}")
        return metadata
    except (Exception, KeyboardInterrupt) as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                         "documents_written": len(lengths), "compressed_bytes_read": budget["used"],
                         "records_seen": records_seen, "skipped_records": dict(skipped)})
        write_json(results / "dataset_failure.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    try:
        prepare(load_config(args.config))
    except (Exception, KeyboardInterrupt) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
