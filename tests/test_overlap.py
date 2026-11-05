from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import filter_datasets, overlap
from src.filter_assets import ROOT
from src.run_filters import STAGES, run_stage
from test_run_results import fixture

# Known removal sets on 8 fixture documents (by index).
REMOVE = {"url": {0, 1, 2}, "language": {1, 2, 3}, "quality": {2, 3, 4, 5}, "dedup": {6}}


class SetFilter:
    assets = {}

    def __init__(self, stage):
        self.reject = {f"fixture-{i}" for i in REMOVE[stage]}
        self.ids = []

    def evaluate(self, document):
        kept = document["document_id"] not in self.reject
        return kept, None if kept else "fixture_rule", {}

    def add(self, document):
        self.ids.append(document["document_id"])

    def decisions(self):
        return {i: (i not in self.reject, None if i not in self.reject else "fixture_duplicate", {}) for i in self.ids}


def words(text):
    return tuple(len(part.split()) for part in text.split(".") if part.split())


def snapshot(directory):
    return {path: path.read_bytes() for path in Path(directory).rglob("*") if path.is_file()}


class OverlapTests(unittest.TestCase):
    def make_run(self, root, mode="independent", count=8):
        baseline = root / "baseline.jsonl"
        baseline.write_text("".join(json.dumps(fixture(i)) + "\n" for i in range(count)), encoding="utf-8")
        execute = lambda stage, config, root: run_stage(stage, config, root=root, factory=lambda *a: SetFilter(stage))
        with patch.object(filter_datasets, "ROOT", root), patch.object(filter_datasets, "run_stage", execute):
            return filter_datasets.run((), STAGES, root / "results", baseline, "fixture",
                                       ROOT / "filter_config.json", mode=mode)[0]

    def test_pair_statistics_match_hand_computation(self):
        patterns = [0] * 8
        for bit, stage in enumerate(STAGES):
            for index in REMOVE[stage]:
                patterns[index] |= 1 << bit
        url_language = overlap.pair_statistics(Counter(patterns), "url", "language")
        self.assertEqual((url_language["both_removed"], url_language["union_removed"]), (2, 4))
        self.assertAlmostEqual(url_language["jaccard"], .5)
        self.assertAlmostEqual(url_language["containment_a_in_b"], 2 / 3)
        self.assertAlmostEqual(url_language["lift"], 2 / (3 * 3 / 8))
        self.assertAlmostEqual(url_language["phi"], (2 * 4 - 1 * 1) / (3 * 3 * 5 * 5) ** .5)
        url_dedup = overlap.pair_statistics(Counter(patterns), "url", "dedup")
        self.assertEqual((url_dedup["jaccard"], url_dedup["lift"]), (0, 0))
        summary = overlap.overlap_summary(patterns, iterations=200, seed=1)
        self.assertEqual(summary["survived_all_filters"], 1)
        self.assertEqual(summary["removed_by_exactly"], {"0": 1, "1": 4, "2": 2, "3": 1, "4": 0})
        exclusive = {row["filter"]: row["removed_only_by_this_filter"] for row in summary["per_filter"]}
        self.assertEqual(exclusive, {"url": 1, "language": 0, "quality": 2, "dedup": 1})
        self.assertEqual(sum(row["documents"] for row in summary["pattern_counts"]), 8)

    def test_bootstrap_is_seeded_and_brackets_estimate(self):
        patterns = [0] * 300 + [1] * 100 + [3] * 80 + [2] * 120
        first = overlap.overlap_summary(patterns, iterations=500, seed=7)
        self.assertEqual(first, overlap.overlap_summary(patterns, iterations=500, seed=7))
        pair = first["pairs"][0]
        self.assertLessEqual(pair["jaccard_ci95_low"], pair["jaccard"])
        self.assertGreaterEqual(pair["jaccard_ci95_high"], pair["jaccard"])
        self.assertIsNone(overlap.overlap_summary(patterns, iterations=0)["pairs"][0]["jaccard_ci95_low"])

    def test_degenerate_pair_has_no_undefined_numbers(self):
        summary = overlap.overlap_summary([0, 0, 0], iterations=50)
        for row in summary["pairs"]:
            self.assertIsNone(row["jaccard"])
            self.assertIsNone(row["phi"])
            self.assertIsNone(row["jaccard_ci95_low"])

    def test_wasserstein_known_answers(self):
        self.assertEqual(overlap.wasserstein({1: 1}, {1: 5}), 0)
        self.assertAlmostEqual(overlap.wasserstein({1: 1}, {4: 1}), 3)
        self.assertAlmostEqual(overlap.wasserstein({1: 1, 3: 1}, {2: 2}), 1)
        self.assertIsNone(overlap.wasserstein({}, {2: 1}))

    def test_end_to_end_run_leaves_sources_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            run = self.make_run(root)
            before = snapshot(root / "results") | {root / "baseline.jsonl": (root / "baseline.jsonl").read_bytes()}
            loaded = overlap.load_run(run)
            self.assertEqual(loaded["ids"], [f"fixture-{i}" for i in range(8)])
            directory = overlap.analyze([run], root / "results", iterations=100, sentence_lengths=words)
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)
            self.assertIn("_cross_filter_overlap_", directory.name)
            target = directory / "fixture"
            for name in ("pairs.csv", "patterns.csv", "per_filter.csv", "profiles.csv", "overlap.json",
                         "overlap_heatmaps.png", "removal_patterns.png", "removed_sentence_cdf.png"):
                self.assertTrue((target / name).is_file(), name)
            result = json.loads((target / "overlap.json").read_text(encoding="utf-8"))
            self.assertEqual(result["survived_all_filters"], 1)
            profiles = {row["group"]: row for row in result["profiles"]}
            self.assertEqual(profiles["baseline"]["documents"], 8)
            self.assertEqual(profiles["quality:removed"]["documents"], 4)
            self.assertEqual(profiles["quality:removed_only_here"]["documents"], 2)
            self.assertEqual(profiles["quality:retained"]["documents"], 4)
            self.assertEqual(profiles["language:removed_only_here"]["documents"], 0)
            self.assertEqual(profiles["survived_all"]["documents"], 1)
            self.assertEqual(json.loads((directory / "summary.json").read_text())["status"], "complete")
            self.assertEqual(overlap.latest_independent_runs(root / "results"), [run])
            second = overlap.analyze([run], root / "results", iterations=100, profiles=False)
            self.assertNotEqual(directory, second)
            self.assertFalse((second / "fixture/profiles.csv").exists())

    def test_rejects_tampered_logs_sequential_runs_and_changed_baseline(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            run = self.make_run(root)
            log = run / "details/results/quality_decisions.jsonl"
            original = log.read_bytes()
            log.write_bytes(original.replace(b'"kept": false', b'"kept": true', 1))
            with self.assertRaisesRegex(ValueError, "recorded hash"):
                overlap.load_run(run)
            log.write_bytes(original)
            baseline = root / "baseline.jsonl"
            baseline.write_text(baseline.read_text().replace("fixture 7", "fixture 77"))
            with self.assertRaisesRegex(ValueError, "baseline differs"):
                overlap.analyze([run], root / "results", iterations=0, sentence_lengths=words)
            sequential = self.make_run(Path(tempfile.mkdtemp(dir=root)), mode="sequential")
            with self.assertRaisesRegex(ValueError, "independent"):
                overlap.load_run(sequential)

    def test_menu_runs_overlap_on_latest_runs_or_explains_absence(self):
        from src import interactive
        runs = [Path("results/2026-01-01_A_independent_00-00-00")]
        for found, expected in ((runs, "Completed: done"), ([], "No complete independent runs")):
            answers, output, analyzer = iter(("3", "0")), [], Mock(return_value="done")
            with patch("src.overlap.latest_independent_runs", return_value=found):
                self.assertEqual(interactive.main(lambda _: next(answers), output.append, Mock(), analyzer), 0)
            self.assertTrue(any(expected in line for line in output))
            self.assertEqual(analyzer.called, bool(found))

    def test_cli_requires_a_source(self):
        with self.assertRaises(SystemExit):
            overlap.main([])


if __name__ == "__main__":
    unittest.main()
