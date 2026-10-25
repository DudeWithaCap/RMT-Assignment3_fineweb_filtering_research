from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.filter_assets import ROOT, sha256_file
from stage4.collect import measure_rows
from stage4.metrics import (Accumulator, MeasurementTokenizer, entropy_bits, histogram_quantile,
                            relative_to_baseline, safe_percentage, sentence_statistics)
from stage4.preserve import OUTPUT, verify_preserved


def row(index, text, domain="example.org"):
    document = {"document_id": f"fixture-{index}", "text": text, "domain": domain,
                "source": {"wet_path": "fixture.warc.wet.gz", "record_index": index}}
    return (json.dumps(document, ensure_ascii=False) + "\n").encode(), document


def writers():
    sentences, documents = io.StringIO(), io.StringIO()
    return (csv.DictWriter(sentences, fieldnames=["dataset", "condition", "document_id", "sentence_index", "sentence_tokens"]),
            csv.DictWriter(documents, fieldnames=["dataset", "condition", "document_id", "domain", "token_count", "sentence_count", "character_count", "source_wet_path", "source_record_index"]),
            sentences, documents)


class FixtureTokenizer:
    def sentence_lengths(self, text):
        return {"fixture A": (2, 4), "fixture B": (3,), "": ()}[text]


class MetricFormulaTests(unittest.TestCase):
    def test_document_token_and_domain_counts(self):
        accumulator = Accumulator()
        accumulator.add("a.example", (2, 4), 10)
        accumulator.add("b.example", (3,), 3)
        accumulator.add("a.example", (), 0)
        metrics = accumulator.summarize()
        self.assertEqual(metrics["document_count"], 3)
        self.assertEqual(metrics["token_count"], 9)
        self.assertEqual(metrics["sentence_count"], 3)
        self.assertEqual(metrics["character_count"], 13)
        self.assertEqual(metrics["unique_domains"], 2)

    def test_known_sentence_statistics(self):
        metrics = sentence_statistics(Counter({2: 1, 4: 2, 8: 1}))
        self.assertEqual(metrics["sentence_count"], 4)
        self.assertEqual(metrics["sentence_mean_tokens"], 4.5)
        self.assertEqual(metrics["sentence_median_tokens"], 4)
        self.assertAlmostEqual(metrics["sentence_std_population_tokens"], math.sqrt(4.75))
        self.assertEqual(metrics["sentence_p25_tokens"], 3.5)
        self.assertEqual(metrics["sentence_p75_tokens"], 5)

    def test_percentile_linear_interpolation_and_bounds(self):
        histogram = Counter({1: 1, 2: 1, 3: 1, 4: 1})
        self.assertEqual(histogram_quantile(histogram, .25), 1.75)
        self.assertEqual(histogram_quantile(histogram, .75), 3.25)
        self.assertEqual(histogram_quantile(histogram, 0), 1)
        self.assertEqual(histogram_quantile(histogram, 1), 4)
        with self.assertRaises(ValueError):
            histogram_quantile(histogram, 1.1)

    def test_entropy_known_distributions(self):
        self.assertEqual(entropy_bits(Counter()), 0)
        self.assertEqual(entropy_bits(Counter({"a": 5})), 0)
        self.assertEqual(entropy_bits(Counter({"a": 2, "b": 2})), 1)
        self.assertEqual(entropy_bits(Counter({str(i): 1 for i in range(4)})), 2)
        self.assertAlmostEqual(entropy_bits(Counter({"a": 3, "b": 1})), 0.8112781244591328)
        with self.assertRaises(ValueError):
            entropy_bits({"a": -1})

    def test_relative_reductions_and_domain_retention(self):
        baseline, current = Accumulator(), Accumulator()
        for domain, lengths in (("a", (2,)), ("a", (3,)), ("b", (1,)), ("b", (4,))):
            baseline.add(domain, lengths)
        current.add("a", (2,))
        current.add("a", (1,))
        result = relative_to_baseline(current.summarize(), baseline.summarize())
        self.assertEqual(result["document_reduction_absolute"], 2)
        self.assertEqual(result["document_reduction_percentage"], 50)
        self.assertEqual(result["token_reduction_absolute"], 7)
        self.assertEqual(result["token_reduction_percentage"], 70)
        self.assertEqual(result["domain_retention_percentage"], 50)
        self.assertEqual(result["domain_entropy_bits_delta_from_baseline"], -1)

    def test_empty_dataset_and_zero_denominators(self):
        empty = Accumulator().summarize()
        relative = relative_to_baseline(empty, empty)
        self.assertEqual(empty["document_count"], 0)
        self.assertEqual(empty["token_count"], 0)
        self.assertEqual(empty["domain_entropy_bits"], 0)
        self.assertIsNone(empty["sentence_mean_tokens"])
        self.assertIsNone(relative["token_reduction_percentage"])
        self.assertIsNone(relative["document_reduction_percentage"])
        self.assertIsNone(relative["domain_retention_percentage"])
        self.assertIsNone(safe_percentage(0, 0))
        json.dumps(relative, allow_nan=False)

    def test_empty_condition_against_nonempty_baseline(self):
        baseline = Accumulator()
        baseline.add("a", (3,))
        result = relative_to_baseline(Accumulator().summarize(), baseline.summarize())
        self.assertEqual(result["document_reduction_percentage"], 100)
        self.assertEqual(result["token_reduction_percentage"], 100)
        self.assertEqual(result["domain_retention_percentage"], 0)

    def test_single_sentence_and_zero_token_document(self):
        metrics = sentence_statistics(Counter({7: 1}))
        self.assertEqual(metrics["sentence_std_population_tokens"], 0)
        self.assertEqual(metrics["sentence_median_tokens"], 7)
        accumulator = Accumulator()
        accumulator.add("a", ())
        self.assertEqual(accumulator.summarize()["document_count"], 1)
        self.assertIsNone(relative_to_baseline(accumulator.summarize(), accumulator.summarize())["token_reduction_percentage"])
        with self.assertRaises(ValueError):
            accumulator.add("a", (0,))


class MeasurementTests(unittest.TestCase):
    def test_same_calculation_for_three_dataset_labels(self):
        results = []
        for dataset in (1, 2, 3):
            sentence_writer, document_writer, _, _ = writers()
            accumulator, _, _, _ = measure_rows([row(1, "fixture A"), row(2, "fixture B"), row(3, "")], FixtureTokenizer(),
                                                 sentence_writer, document_writer, dataset, "baseline")
            results.append(accumulator.summarize())
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        self.assertEqual(results[0]["token_count"], 9)

    def test_cache_preserves_counts_and_rejects_changed_document(self):
        sw, dw, _, _ = writers()
        original = row(1, "fixture A")
        _, cache, _, _ = measure_rows([original], FixtureTokenizer(), sw, dw, 1, "baseline")
        actual, _, _, _ = measure_rows([original], FixtureTokenizer(), sw, dw, 1, "url", cache)
        self.assertEqual(actual.summarize()["token_count"], 6)
        with self.assertRaisesRegex(ValueError, "not identical"):
            measure_rows([row(1, "fixture B")], FixtureTokenizer(), sw, dw, 1, "url", cache)

    def test_same_inputs_reproduce_numeric_and_intermediate_output(self):
        outputs = []
        for _ in range(2):
            sw, dw, sentences, documents = writers()
            accumulator, _, _, _ = measure_rows([row(1, "fixture A"), row(2, "fixture B")], FixtureTokenizer(), sw, dw, 1, "baseline")
            outputs.append((accumulator.summarize(), sentences.getvalue(), documents.getvalue()))
        self.assertEqual(outputs[0], outputs[1])

    def test_supplementary_transformation_does_not_mutate_documents(self):
        class FixtureTransform:
            def filter(self, working):
                working.text = "fixture B"
                return True

        original = row(1, "fixture A")
        original_text = original[1]["text"]
        sw, dw, _, _ = writers()
        _, cache, _, _ = measure_rows([original], FixtureTokenizer(), sw, dw, 1, "baseline")
        result, _, _, changed = measure_rows([original], FixtureTokenizer(), sw, dw, 1, "quality_c4_transformed", cache, FixtureTransform())
        self.assertEqual(result.summarize()["document_count"], 1)
        self.assertEqual(result.summarize()["token_count"], 3)
        self.assertEqual(changed, 1)
        self.assertEqual(original[1]["text"], original_text)

    def test_preservation_guard_detects_fixture_mutation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            protected = root / "original.txt"
            protected.write_text("original", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": {"original.txt": {"sha256": sha256_file(protected)}}}), encoding="utf-8")
            with patch("stage4.preserve.ROOT", root), patch("stage4.preserve.MANIFEST", manifest):
                self.assertEqual(verify_preserved(), 1)
                protected.write_text("changed fixture", encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    verify_preserved()


@unittest.skipUnless((ROOT / "data/assets/punkt_tab.zip").exists(), "Local tokenizer asset is required; no network fetch during tests")
class LocalTokenizerTests(unittest.TestCase):
    def test_actual_tokenizer_matches_existing_implementation(self):
        tokenizer = MeasurementTokenizer(ROOT / "data/assets")
        for text in ("Hello world. This is a test!", "Привет, мир! Это тест.", "你好世界。再次见面。", "", "\n\t "):
            self.assertEqual(sum(tokenizer.sentence_lengths(text)), len(tokenizer.tokenizer.word_tokenize(text)))
        self.assertEqual(tokenizer.sentence_lengths("Hello world. This is a test!"), (3, 5))


@unittest.skipUnless((OUTPUT / "metrics.json").exists(), "Stage 4 results not yet collected")
class ActualResultsTests(unittest.TestCase):
    def test_all_conditions_reconcile_with_exported_intermediates(self):
        import numpy as np

        metrics = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
        extras = json.loads((OUTPUT / "quality_c4_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(len(metrics), 15)
        self.assertEqual(len(extras), 3)
        documents, domains, tokens, histograms = Counter(), defaultdict(Counter), Counter(), defaultdict(Counter)
        with (OUTPUT / "document_metrics.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (int(row["dataset"]), row["condition"])
                documents[key] += 1
                tokens[key] += int(row["token_count"])
                domains[key][row["domain"]] += 1
        with (OUTPUT / "sentence_lengths.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                histograms[(int(row["dataset"]), row["condition"])][int(row["sentence_tokens"])] += 1
        baselines = {row["dataset"]: row for row in metrics if row["condition"] == "baseline"}
        for result in metrics + extras:
            key = (result["dataset"], result["condition"])
            with self.subTest(key=key):
                self.assertEqual(result["document_count"], documents[key])
                self.assertEqual(result["token_count"], tokens[key])
                values = np.repeat(list(histograms[key]), list(histograms[key].values()))
                self.assertEqual(result["sentence_count"], len(values))
                self.assertEqual(result["token_count"], int(values.sum()))
                self.assertAlmostEqual(result["sentence_mean_tokens"], float(values.mean()), places=9)
                self.assertAlmostEqual(result["sentence_std_population_tokens"], float(values.std(ddof=0)), places=9)
                for field, percentile in (("sentence_median_tokens", 50), ("sentence_p25_tokens", 25), ("sentence_p75_tokens", 75)):
                    self.assertAlmostEqual(result[field], float(np.percentile(values, percentile, method="linear")), places=9)
                expected_entropy = -sum(count / documents[key] * math.log2(count / documents[key]) for count in domains[key].values())
                self.assertAlmostEqual(result["domain_entropy_bits"], expected_entropy, places=9)
                self.assertEqual(result["unique_domains"], len(domains[key]))
                baseline = baselines[result["dataset"]]
                self.assertAlmostEqual(result["token_reduction_percentage"], 100 * (baseline["token_count"] - result["token_count"]) / baseline["token_count"])
                self.assertAlmostEqual(result["domain_retention_percentage"], 100 * result["unique_domains"] / baseline["unique_domains"])


if __name__ == "__main__":
    unittest.main()
