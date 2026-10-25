from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.filter_assets import ROOT, sha256_file
from src.run_filters import STAGES, load_config, run_stage
from stage35.common import check_legacy, filter_settings
from stage35.prepare import distinct_segment_paths, priority, select_window
from stage35.run import validate_stage
from stage35.summarize import fingerprints, pair_overlap


def fixture_row(index, text="Explicit automated test fixture text."):
    return {"document_id": f"fixture-{index}", "text": text, "url": f"https://example.org/{index}",
            "domain": "example.org", "character_count": len(text), "original_character_count": len(text), "truncated": False,
            "source": {"wet_path": "fixture.warc.wet.gz", "record_index": index, "warc_record_id": f"urn:fixture:{index}"}}


class Stage35Tests(unittest.TestCase):
    def test_hash_sampling_matches_full_sort_and_replays(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "candidates.jsonl"
            rows = [fixture_row(i) for i in range(100)]
            path.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows))
            first, count = select_window(path, 10, 42)
            expected = sorted(sorted(rows, key=lambda row: priority(row["document_id"], 42))[:10], key=lambda row: row["source"]["record_index"])
            self.assertEqual(count, 100)
            self.assertEqual([json.loads(line) for line in first], expected)
            self.assertEqual(first, select_window(path, 10, 42)[0])
            self.assertNotEqual(first, select_window(path, 10, 43)[0])
            self.assertNotEqual([row["document_id"] for row in expected], [row["document_id"] for row in rows[:10]])

    def test_insufficient_candidates_are_an_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "candidates.jsonl"
            path.write_text(json.dumps(fixture_row(0)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Only 1"):
                select_window(path, 2, 42)

    def test_distinct_segments_and_archive_exclusion(self):
        paths = [f"crawl-data/CC-MAIN-2025-18/segments/{segment}/wet/{name}.warc.wet.gz"
                 for segment, name in (("s1", "a"), ("s1", "b"), ("s2", "c"), ("s3", "d"))]
        selected = distinct_segment_paths(paths, 42, 2, excluded=[paths[0]])
        self.assertEqual(len(selected), 2)
        self.assertNotIn(paths[0], selected)
        self.assertEqual(len({path.split("/")[3] for path in selected}), 2)
        self.assertEqual(selected, distinct_segment_paths(list(reversed(paths)), 42, 2, excluded=[paths[0]]))

    def test_dataset_adapter_preserves_all_filter_thresholds(self):
        original = load_config(ROOT / "filter_config.json")
        before = deepcopy(original)
        with patch("stage35.common.load_config", return_value=original), patch("stage35.common.sha256_file", return_value="a" * 64):
            settings = filter_settings(2)
        self.assertEqual(original, before)
        for stage in STAGES:
            self.assertEqual(settings[stage], original[stage])
        self.assertTrue(settings["baseline"].endswith("baseline_2.jsonl"))
        self.assertEqual(settings["max_text_characters"], original["max_text_characters"])
        self.assertEqual(settings["baseline_sha256"], "a" * 64)

    def test_preservation_guard_detects_mutation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            protected = root / "protected.txt"
            protected.write_text("original", encoding="utf-8")
            (root / "legacy_snapshot.json").write_text(json.dumps({"files": {"protected.txt": {"sha256": sha256_file(protected)}}}), encoding="utf-8")
            with patch("stage35.common.ROOT", root), patch("stage35.common.EVIDENCE", root):
                self.assertEqual(check_legacy(), 1)
                protected.write_text("changed fixture", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    check_legacy()

    def test_overlap_reports_duplicates_without_removal(self):
        a = [fingerprints(fixture_row(1)), fingerprints(fixture_row(2))]
        b = [fingerprints(fixture_row(3)), fingerprints(fixture_row(4))]
        before = deepcopy((a, b))
        overlap = {row["metric"]: row for row in pair_overlap(a, b)}
        self.assertEqual(overlap["document_id"]["matching_record_pairs"], 0)
        self.assertEqual(overlap["exact_capped_text"]["shared_unique_values"], 1)
        self.assertEqual(overlap["exact_capped_text"]["matching_record_pairs"], 4)
        self.assertEqual((a, b), before)

    def test_isolated_run_and_source_preservation_verification(self):
        class KeepAll:
            assets = {}
            def evaluate(self, document):
                return True, None, {}

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            baseline = root / "baseline_2.jsonl"
            baseline.write_text("".join(json.dumps(fixture_row(i)) + "\n" for i in range(2)), encoding="utf-8")
            original_bytes = baseline.read_bytes()
            config = load_config(ROOT / "filter_config.json")
            config.update(baseline=str(baseline), baseline_sha256=sha256_file(baseline), expected_documents=2)
            run_stage("url", config, root=root / "isolated", factory=lambda *args: KeepAll())
            validate_stage("url", config, root / "isolated")
            self.assertEqual(baseline.read_bytes(), original_bytes)
            wrong_config = deepcopy(config)
            wrong_config["baseline_sha256"] = "a" * 64
            with self.assertRaises(AssertionError):
                validate_stage("url", wrong_config, root / "isolated")
            output = root / "isolated/data/processed/filtered_url.jsonl"
            with output.open(encoding="utf-8") as stream:
                changed = [json.loads(line) for line in stream]
            changed[0]["source"]["warc_record_id"] = "changed fixture"
            output.write_text("".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8")
            with self.assertRaises(AssertionError):
                validate_stage("url", config, root / "isolated")


if __name__ == "__main__":
    unittest.main()
