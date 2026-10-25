import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from src.prepare_dataset import (ROOT, BudgetReader, document_from_record,
                                 load_config, ordered_paths, prepare, read_text,
                                 validate_document)


def wet_fixture():
    output = io.BytesIO()
    writer = WARCWriter(output, gzip=True)
    for text in ("Test fixture: multilingual \u4f60\u597d \U0001f600 " * 1000, ""):
        record = writer.create_warc_record(
            "https://Example.org/path", "conversion", payload=io.BytesIO(text.encode()),
            warc_headers_dict={"Content-Type": "text/plain"})
        writer.write_record(record)
        record.raw_stream.close()
    return output.getvalue()


class Response:
    def __init__(self, content):
        self.raw = io.BytesIO(content)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()

    def raise_for_status(self):
        pass


class PreparationTests(unittest.TestCase):
    def test_config_and_sampling(self):
        config = load_config(ROOT / "config.json")
        self.assertEqual(config["dataset_size"], 1000)
        self.assertEqual(config["max_text_characters"], 20000)
        self.assertEqual(ordered_paths(["b", "a", "c"], 42),
                         ordered_paths(["c", "b", "a"], 42))

    def test_incremental_unicode_and_truncation(self):
        text = "\u4f60\u597d\U0001f600" * 50000
        prefix, count = read_text(io.BytesIO(text.encode()), 20000)
        self.assertEqual(prefix, text[:20000])
        self.assertEqual(count, len(text))

    def test_conversion_structure_and_empty_preservation(self):
        records = ArchiveIterator(io.BytesIO(wet_fixture()))
        document, reason = document_from_record(next(records), "fixture.warc.wet.gz", 0, 20000)
        self.assertIsNone(reason)
        validate_document(document)
        self.assertEqual(document["domain"], "example.org")
        self.assertEqual(document["url"], "https://Example.org/path")
        self.assertTrue(document["truncated"])
        self.assertEqual(len(document["text"]), 20000)
        document["character_count"] = -1
        with self.assertRaises(ValueError):
            validate_document(document)
        empty, reason = document_from_record(next(records), "fixture.warc.wet.gz", 1, 20000)
        self.assertIsNone(reason)
        self.assertEqual(empty["text"], "")

    def test_missing_url_is_accounted_for(self):
        record = next(ArchiveIterator(io.BytesIO(wet_fixture())))
        record.rec_headers.replace_header("WARC-Target-URI", "invalid")
        document, reason = document_from_record(record, "fixture", 0, 20000)
        self.assertIsNone(document)
        self.assertEqual(reason, "missing_or_invalid_url")

    def test_download_budget(self):
        budget = {"limit": 5, "used": 0}
        reader = BudgetReader(io.BytesIO(b"123456789"), budget)
        self.assertEqual(reader.read(100), b"12345")
        with self.assertRaises(RuntimeError):
            reader.read(1)
        self.assertEqual(budget["used"], 5)

    def test_offline_pipeline_success_and_incomplete_failure(self):
        config = load_config(ROOT / "config.json")
        config["dataset_size"] = 2
        config["max_source_files"] = 1
        path = f"crawl-data/{config['crawl']}/segments/fixture/wet/fixture.warc.wet.gz"
        manifest = gzip.compress((path + "\n").encode())
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            with patch("src.prepare_dataset.requests.Session") as factory:
                session = factory.return_value.__enter__.return_value
                session.get.side_effect = [Response(manifest), Response(wet_fixture())]
                metadata = prepare(config, root)
            baseline = root / "data/processed/baseline.jsonl"
            documents = [json.loads(line) for line in baseline.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(documents), 2)
            self.assertEqual(len({d["document_id"] for d in documents}), 2)
            self.assertEqual(metadata["statistics"]["empty_documents"], 1)
            self.assertEqual(metadata["dataset_sha256"], hashlib.sha256(baseline.read_bytes()).hexdigest())
            self.assertEqual(metadata["statistics"]["total_characters"], sum(len(d["text"]) for d in documents))
            with self.assertRaises(FileExistsError):
                prepare(config, root)
        config["dataset_size"] = 3
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            with patch("src.prepare_dataset.requests.Session") as factory:
                factory.return_value.__enter__.return_value.get.side_effect = [Response(manifest), Response(wet_fixture())]
                with self.assertRaisesRegex(RuntimeError, "Only 2"):
                    prepare(config, root)
            self.assertFalse((root / "data/processed/baseline.jsonl").exists())
            report = json.loads((root / "results/dataset_failure.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")


class AcquiredDatasetTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "data/processed/baseline.jsonl").exists(),
                         "Real baseline not acquired; offline tests still run")
    def test_real_baseline_structure_and_metadata(self):
        metadata = json.loads((ROOT / "results/dataset_metadata.json").read_text(encoding="utf-8"))
        config = metadata["config"]
        seen, digest, count, characters = set(), hashlib.sha256(), 0, 0
        with (ROOT / "data/processed/baseline.jsonl").open("rb") as stream:
            for line in stream:
                digest.update(line)
                document = json.loads(line)
                validate_document(document, config["max_text_characters"])
                self.assertNotIn(document["document_id"], seen)
                seen.add(document["document_id"])
                count += 1
                characters += len(document["text"])
        self.assertEqual(count, config["dataset_size"])
        self.assertEqual(count, metadata["statistics"]["document_count"])
        self.assertEqual(characters, metadata["statistics"]["total_characters"])
        self.assertEqual(digest.hexdigest(), metadata["dataset_sha256"])


if __name__ == "__main__":
    unittest.main()
