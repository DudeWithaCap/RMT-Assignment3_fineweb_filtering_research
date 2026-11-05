import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from warcio.warcwriter import WARCWriter

from src.datasets.prepare_large import prepare_large
from src.filter_assets import ROOT
from src.prepare_dataset import BASE_URL, load_config, ordered_paths, prepare
from test_prepare_dataset import Response


def wet(texts, host):
    output = io.BytesIO()
    writer = WARCWriter(output, gzip=True)
    for index, text in enumerate(texts):
        record = writer.create_warc_record(f"https://{host}/{index}", "conversion",
                                           payload=io.BytesIO(text.encode()),
                                           warc_headers_dict={"Content-Type": "text/plain"})
        writer.write_record(record)
        record.raw_stream.close()
    return output.getvalue()


class FakeSession:
    def __init__(self, files, failures=None):
        self.files, self.failures, self.requests = files, dict(failures or {}), []

    def get(self, url, stream=True, timeout=None):
        self.requests.append(url)
        if self.failures.get(url, 0):
            self.failures[url] -= 1
            raise ConnectionError("503 Service Unavailable")
        return Response(self.files[url])


class LargeAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")
        self.config.update(dataset_size=7, max_source_files=3)
        crawl = self.config["crawl"]
        paths = [f"crawl-data/{crawl}/segments/s{i}/wet/f{i}.warc.wet.gz" for i in range(3)]
        self.ordered = ordered_paths(paths, self.config["sampling_seed"])
        self.files = {BASE_URL + f"crawl-data/{crawl}/wet.paths.gz": gzip.compress("\n".join(paths).encode())}
        for number, path in enumerate(self.ordered):
            self.files[BASE_URL + path] = wet([f"file {number} doc {i}" for i in range(4)], f"site{number}.org")

    def test_resumes_after_outage_and_matches_original_procedure(self):
        second = BASE_URL + self.ordered[1]
        quiet = dict(log=lambda _: None, sleep=lambda _: None)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            with self.assertRaises(ConnectionError):
                prepare_large(self.config, "baseline_big.jsonl", root, FakeSession(self.files, {second: 9}),
                              file_attempts=2, **quiet)
            self.assertFalse((root / "data/processed/baseline_big.jsonl").exists())
            self.assertTrue((root / "data/raw/baseline_big/parts/0000.jsonl").exists())
            session = FakeSession(self.files, {second: 1})
            metadata = prepare_large(self.config, "baseline_big.jsonl", root, session, file_attempts=3, **quiet)
            self.assertNotIn(BASE_URL + self.ordered[0], session.requests)  # cached part reused
            self.assertEqual(session.requests.count(second), 2)  # one outage, one success
            large = (root / "data/processed/baseline_big.jsonl").read_bytes()
            self.assertEqual(len(large.splitlines()), 7)
            self.assertEqual(metadata["statistics"]["document_count"], 7)
            self.assertEqual(metadata["statistics"]["unique_domains"], 2)
            with self.assertRaises(FileExistsError):
                prepare_large(self.config, "baseline_big.jsonl", root, session, **quiet)
            with self.assertRaisesRegex(ValueError, "never replaced"):
                prepare_large(self.config, "baseline.jsonl", root, session, **quiet)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            with patch("src.prepare_dataset.requests.Session") as factory:
                session = factory.return_value.__enter__.return_value
                session.get.side_effect = lambda url, **_: Response(self.files[url])
                prepare(self.config, root)
            original = (root / "data/processed/baseline.jsonl").read_bytes()
        self.assertEqual(large, original)  # same procedure, byte-identical documents

    def test_invalid_record_is_counted_not_fatal(self):
        from src.datasets import prepare_large as module
        original = module.document_from_record

        def flaky(record, path, index, limit):
            if index == 1:
                raise ValueError("Domain does not match source URL")
            return original(record, path, index, limit)

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory, \
                patch.object(module, "document_from_record", flaky):
            metadata = prepare_large(self.config, "baseline_big.jsonl", Path(directory), FakeSession(self.files),
                                     log=lambda _: None)
        first = metadata["files"][0]
        self.assertEqual(first["documents"], 3)
        self.assertEqual(first["skipped_records"], {"invalid_document: Domain does not match source URL": 1})
        self.assertEqual(first["invalid_examples"][0]["url"], "https://site0.org/1")

    def test_insufficient_documents_publish_nothing(self):
        self.config["dataset_size"] = 50
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Only 12"):
                prepare_large(self.config, "baseline_big.jsonl", root, FakeSession(self.files), log=lambda _: None)
            self.assertFalse((root / "data/processed/baseline_big.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
