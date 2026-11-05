from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import filter_datasets
from src.filter_assets import ROOT, sha256_file
from src.interactive import main
from src.run_filters import STAGES, run_stage
from src.run_results import publish
from test_run_results import fixture


class FixtureFilter:
    assets = {}

    def __init__(self, stage):
        self.reject_id = f"fixture-{STAGES.index(stage)}"
        self.ids = []

    def evaluate(self, document):
        kept = document["document_id"] != self.reject_id
        return kept, None if kept else "fixture_rule", {}

    def add(self, document):
        self.ids.append(document["document_id"])

    def decisions(self):
        return {identity: (index == 0, None if index == 0 else "fixture_duplicate", {})
                for index, identity in enumerate(self.ids)}


class SequentialTests(unittest.TestCase):
    def baseline(self, root, count=5):
        path = root / "baseline.jsonl"
        path.write_text("".join(json.dumps(fixture(index)) + "\n" for index in range(count)), encoding="utf-8")
        return path

    def test_chain_final_json_and_repeated_runs_preserve_baseline(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            baseline = self.baseline(root)
            original = baseline.read_bytes()
            calls = []

            def execute(stage, config, root):
                with Path(config["baseline"]).open(encoding="utf-8") as stream:
                    documents = [json.loads(line) for line in stream]
                self.assertEqual(config["expected_documents"], len(documents))
                self.assertEqual(config["baseline_sha256"], sha256_file(Path(config["baseline"])))
                calls.append([row["document_id"] for row in documents])
                return run_stage(stage, config, root=root, factory=lambda *args: FixtureFilter(stage))

            with patch.object(filter_datasets, "ROOT", root), patch.object(filter_datasets, "run_stage", execute):
                first = filter_datasets.run((), STAGES, root / "results", baseline, "fixture", ROOT / "filter_config.json", mode="sequential")[0]
                first_json = (first / "cleaned.json").read_bytes()
                second = filter_datasets.run((), STAGES, root / "results", baseline, "fixture", ROOT / "filter_config.json",
                                             mode="sequential", keep_intermediate=True)[0]
            self.assertEqual([len(ids) for ids in calls], [5, 4, 3, 2] * 2)
            self.assertEqual(calls[1], [f"fixture-{i}" for i in range(1, 5)])
            self.assertEqual(calls[2], [f"fixture-{i}" for i in range(2, 5)])
            self.assertEqual(json.loads(first_json), [fixture(3)])
            self.assertEqual((second / "cleaned.json").read_bytes(), first_json)
            self.assertEqual((first / "cleaned.json").read_bytes(), first_json)
            self.assertEqual(baseline.read_bytes(), original)
            self.assertIn("_sequential_", first.name)
            self.assertFalse((first / "datasets").exists())
            self.assertEqual(len(list((second / "datasets").glob("*.json"))), 4)
            summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["filter_mode"], "sequential")
            self.assertEqual(summary["final_documents"], 1)
            self.assertEqual([row["removed"] for row in summary["filters"]], [1, 1, 1, 1])
            self.assertEqual([row["retained_percent_of_baseline"] for row in summary["filters"]], [80, 60, 40, 20])

    def test_empty_intermediate_skips_later_models_and_exports_empty_json(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            baseline = self.baseline(root, count=2)
            factory_calls = []

            def factory(stage, *args):
                factory_calls.append(stage)
                self.assertEqual(stage, "url")
                evaluator = FixtureFilter(stage)
                evaluator.evaluate = lambda document: (False, "fixture_reject", {})
                return evaluator

            def execute(stage, config, root):
                return run_stage(stage, config, root=root, factory=factory)

            with patch.object(filter_datasets, "ROOT", root), patch.object(filter_datasets, "run_stage", execute):
                directory = filter_datasets.run((), STAGES, root / "results", baseline, "fixture", ROOT / "filter_config.json", mode="sequential")[0]
            self.assertEqual(factory_calls, ["url"])
            self.assertEqual(json.loads((directory / "cleaned.json").read_text()), [])
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual([row["input"] for row in summary["filters"]], [2, 0, 0, 0])
            self.assertTrue(all(row["execution"] == "skipped_empty_input" for row in summary["filters"][1:]))
            self.assertIsNone(summary["filters"][1]["retained_percent"])

    def test_sequential_rejects_partial_filter_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            with patch.object(filter_datasets, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "all four"):
                    filter_datasets.run((), ("url",), root / "results", mode="sequential")
            self.assertFalse((root / "results").exists())

    def test_export_rejects_incorrect_stage_input_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            baseline = self.baseline(root)

            def execute(stage, config, root):
                return run_stage(stage, config, root=root, factory=lambda *args: FixtureFilter(stage))

            with patch.object(filter_datasets, "ROOT", root), patch.object(filter_datasets, "run_stage", execute):
                directory = filter_datasets.run((), STAGES, root / "results", baseline, "fixture", ROOT / "filter_config.json", mode="sequential")[0]
            outputs = {stage: {"dataset": directory / f"details/data/processed/filtered_{stage}.jsonl",
                               "summary": directory / f"details/results/{stage}_summary.json"} for stage in STAGES}
            report_path = outputs["quality"]["summary"]
            report = json.loads(report_path.read_text())
            report["baseline_sha256"] = "f" * 64
            report_path.write_text(json.dumps(report))
            new = root / "verification"
            new.mkdir()
            with self.assertRaisesRegex(ValueError, "Source verification"):
                publish(new, "fixture", baseline, sha256_file(baseline), datetime.now(timezone.utc), outputs,
                        origin="fixture", mode="sequential")


class InteractiveTests(unittest.TestCase):
    def test_invalid_choices_retry_and_sequential_mode_is_selected(self):
        answers = iter(("bad", "2", "bad", "3", "0"))
        output, runner = [], Mock(return_value=[])
        self.assertEqual(main(lambda _: next(answers), output.append, runner), 0)
        self.assertEqual(runner.call_args.args[:2], ((3,), STAGES))
        self.assertEqual(runner.call_args.kwargs["mode"], "sequential")
        self.assertEqual(sum("Invalid choice" in line for line in output), 2)

    def test_custom_path_with_spaces_and_quotes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests", prefix="menu fixture ") as temporary:
            baseline = Path(temporary) / "custom baseline.jsonl"
            baseline.write_text(json.dumps(fixture(0)) + "\n")
            answers = iter(("1", "5", "missing-file.jsonl", f'"{baseline}"', "custom_fixture", "0"))
            output, runner = [], Mock(return_value=[])
            self.assertEqual(main(lambda _: next(answers), output.append, runner), 0)
            self.assertEqual(runner.call_args.kwargs["baseline"], baseline)
            self.assertEqual(runner.call_args.kwargs["dataset_name"], "custom_fixture")
            self.assertEqual(runner.call_args.kwargs["mode"], "independent")
            self.assertTrue(any("File not found" in line for line in output))

    def test_back_and_exit_do_not_start_a_run(self):
        answers = iter(("2", "0", "0"))
        runner = Mock()
        self.assertEqual(main(lambda _: next(answers), lambda _: None, runner), 0)
        runner.assert_not_called()

    def test_error_returns_to_menu_and_eof_exits(self):
        answers = iter(("1", "1", "0"))
        output = []
        self.assertEqual(main(lambda _: next(answers), output.append, Mock(side_effect=FileNotFoundError("missing baseline"))), 0)
        self.assertTrue(any("Run failed: missing baseline" in line for line in output))
        self.assertEqual(main(Mock(side_effect=EOFError), lambda _: None, Mock()), 0)
