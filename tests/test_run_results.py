from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import filter_datasets
from src.filter_assets import ROOT, sha256_file
from src.run_filters import STAGES
from src.run_results import create_run, export_json, inspect_baseline


def fixture(index):
    text = f"Offline test fixture {index}: Привет 世界."
    return {"document_id": f"fixture-{index}", "text": text, "url": f"https://example.org/{index}",
            "domain": "example.org", "character_count": len(text), "original_character_count": len(text),
            "truncated": False, "source": {"wet_path": "fixture.warc.wet.gz", "record_index": index,
                                          "warc_record_id": f"urn:fixture:{index}"}}


class RunResultsTests(unittest.TestCase):
    def test_timezone_date_and_same_instant_do_not_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            parent = Path(temporary)
            started = datetime(2030, 1, 1, 22, 0, tzinfo=timezone.utc)
            first = create_run(parent, "CC_2025_30", started)
            marker = first / "keep.txt"
            marker.write_text("keep")
            second = create_run(parent, "CC_2025_30", started)
            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("2030-01-02_CC_2025_30_03-00-00"))
            self.assertEqual(marker.read_text(), "keep")
            with self.assertRaises(ValueError):
                create_run(parent, "../escape", started)

    def test_json_export_preserves_unicode_metadata_and_empty_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            documents = [fixture(0), fixture(1)]
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents), encoding="utf-8")
            before = source.read_bytes()
            self.assertEqual(export_json(source, root / "cleaned.json"), 2)
            self.assertEqual(json.loads((root / "cleaned.json").read_text(encoding="utf-8")), documents)
            self.assertEqual(source.read_bytes(), before)
            source.write_text("")
            self.assertEqual(export_json(source, root / "empty.json"), 0)
            self.assertEqual(json.loads((root / "empty.json").read_text()), [])

    def test_inspection_rejects_empty_baseline(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            baseline = Path(temporary) / "baseline.jsonl"
            baseline.write_text("")
            with self.assertRaisesRegex(ValueError, "at least one"):
                inspect_baseline(baseline, 20000)

    def test_independent_runs_export_counts_and_rerun_safely(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            baseline = root / "baseline.jsonl"
            documents = [fixture(0), fixture(1)]
            baseline.write_text("".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8")
            before = baseline.read_bytes()
            calls = []

            def fake_stage(stage, config, root):
                incoming = Path(config["baseline"]).read_bytes()
                calls.append((stage, incoming))
                chosen = {"url": [documents[0]], "language": [documents[1]], "quality": [], "dedup": documents}[stage]
                data, results = root / "data/processed", root / "results"
                data.mkdir(parents=True, exist_ok=True)
                results.mkdir(parents=True, exist_ok=True)
                output = data / f"filtered_{stage}.jsonl"
                output.write_text("".join(json.dumps(row) + "\n" for row in chosen), encoding="utf-8")
                summary = {"status": "complete", "baseline_sha256": config["baseline_sha256"],
                           "baseline_sha256_after": config["baseline_sha256"], "output_sha256": sha256_file(output),
                           "input_count": 2, "output_count": len(chosen), "removed_count": 2 - len(chosen),
                           "removal_reasons": {"fixture_rule": 2 - len(chosen)}}
                (results / f"{stage}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                return summary

            with patch.object(filter_datasets, "ROOT", root), patch.object(filter_datasets, "run_stage", fake_stage):
                first = filter_datasets.run((), STAGES, root / "results", baseline, "fixture_dataset", ROOT / "filter_config.json")[0]
                saved = (first / "summary.json").read_bytes()
                second = filter_datasets.run((), STAGES, root / "results", baseline, "fixture_dataset", ROOT / "filter_config.json")[0]
            self.assertNotEqual(first, second)
            self.assertEqual((first / "summary.json").read_bytes(), saved)
            self.assertEqual(baseline.read_bytes(), before)
            self.assertTrue(all(incoming == before for _, incoming in calls))
            summary = json.loads(saved)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual([row["retained"] for row in summary["filters"]], [1, 1, 0, 2])
            self.assertEqual(json.loads((first / "datasets/language.json").read_text(encoding="utf-8")), [documents[1]])
            self.assertEqual(json.loads((first / "datasets/quality.json").read_text()), [])
            self.assertTrue((first / "retention.png").read_bytes().startswith(b"\x89PNG"))

    def test_failed_filter_is_recorded_without_success_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            baseline = root / "baseline.jsonl"
            baseline.write_text(json.dumps(fixture(0)) + "\n")
            with patch.object(filter_datasets, "ROOT", root), patch.object(filter_datasets, "run_stage", side_effect=RuntimeError("missing asset")):
                with self.assertRaisesRegex(RuntimeError, "missing asset"):
                    filter_datasets.run((), ("url",), root / "results", baseline, "fixture_dataset", ROOT / "filter_config.json")
            summary = json.loads(next((root / "results").glob("*/summary.json")).read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertIn("missing asset", summary["error"])
            self.assertFalse(list((root / "results").glob("*/retention.png")))

