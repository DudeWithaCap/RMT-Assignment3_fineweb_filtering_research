from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.filter_assets import ROOT, sha256_file
from stage4.metrics import entropy_bits
from stage5.analyze import document_mean_lengths
from stage5.preserve import OUTPUT, verify_preserved
from stage5.statistics import (descriptive, distribution_distances, expected_entropy_same_size,
                               expected_unique_domains, missing_probability, retention_discordance)


class EffectSizeTests(unittest.TestCase):
    def test_known_cdf_and_wasserstein_distances(self):
        result = distribution_distances({1: 1, 3: 1}, {2: 2})
        self.assertEqual(result["cdf_sup_distance"], .5)
        self.assertEqual(result["wasserstein1_tokens"], 1)
        shifted = distribution_distances({1: 3}, {4: 5})
        self.assertEqual(shifted["cdf_sup_distance"], 1)
        self.assertEqual(shifted["wasserstein1_tokens"], 3)

    def test_distance_identity_symmetry_and_scaling(self):
        a, b = {1: 3, 5: 2}, {2: 4, 4: 1}
        self.assertEqual(distribution_distances(a, a), {"cdf_sup_distance": 0, "wasserstein1_tokens": 0})
        for key, value in distribution_distances(a, b).items():
            self.assertAlmostEqual(value, distribution_distances(b, a)[key])
            self.assertAlmostEqual(value, distribution_distances({x: n * 5 for x, n in a.items()}, b)[key])

    def test_empty_distributions_are_undefined(self):
        self.assertEqual(distribution_distances({}, {2: 1}), {"cdf_sup_distance": None, "wasserstein1_tokens": None})

    def test_hypergeometric_missing_probability(self):
        self.assertAlmostEqual(missing_probability(4, 2, 2), 1 / 6)
        self.assertEqual(missing_probability(4, 2, 4), 0)
        self.assertEqual(missing_probability(4, 2, 0), 1)
        with self.assertRaises(ValueError):
            missing_probability(4, 2, 5)

    def test_known_same_size_domain_expectations(self):
        domains = {"a": 2, "b": 2}
        self.assertAlmostEqual(expected_unique_domains(domains, 2), 5 / 3)
        self.assertAlmostEqual(expected_entropy_same_size(domains, 2), 2 / 3)
        self.assertEqual(expected_unique_domains({}, 0), 0)
        self.assertEqual(expected_entropy_same_size({}, 0), 0)

    def test_domain_expectations_match_exhaustive_finite_subsets(self):
        population = ["a", "a", "a", "b", "b", "c"]
        domains = Counter(population)
        for k in range(len(population) + 1):
            subsets = list(combinations(range(len(population)), k))
            observed = [Counter(population[i] for i in subset) for subset in subsets]
            richness = sum(len(counts) for counts in observed) / len(observed)
            entropy = sum(entropy_bits(counts) for counts in observed) / len(observed)
            self.assertAlmostEqual(expected_unique_domains(domains, k), richness, places=12)
            self.assertAlmostEqual(expected_entropy_same_size(domains, k), entropy, places=12)

    def test_one_record_per_domain_does_not_imply_excess_diversity_loss(self):
        domains = {str(i): 1 for i in range(100)}
        self.assertAlmostEqual(expected_unique_domains(domains, 10), 10)
        self.assertAlmostEqual(expected_entropy_same_size(domains, 10), math.log2(10))

    def test_invalid_domain_counts(self):
        with self.assertRaises(ValueError):
            expected_unique_domains({"a": -1}, 0)
        with self.assertRaises(ValueError):
            expected_entropy_same_size({"a": 4}, 5)

    def test_paired_discordance_not_independent_samples(self):
        result = retention_discordance({"a", "b"}, {"b", "c"}, {"a", "b", "c", "d"})
        self.assertEqual(result["removed_by_both"], 1)
        self.assertEqual(result["removed_by_first_only"], 1)
        self.assertEqual(result["removed_by_second_only"], 1)
        self.assertEqual(result["kept_by_both"], 1)
        self.assertEqual(result["document_reduction_difference_pp"], 0)
        with self.assertRaises(ValueError):
            retention_discordance({"unknown"}, set(), {"a"})

    def test_document_weighting_differs_from_sentence_pooling(self):
        docs = {"a": {"token_count": 1000, "sentence_count": 100},
                "b": {"token_count": 100, "sentence_count": 1},
                "c": {"token_count": 0, "sentence_count": 0}}
        result = descriptive(document_mean_lengths(docs))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mean"], 55)
        self.assertNotEqual(result["mean"], 1100 / 101)

    def test_cross_dataset_descriptives_are_not_inference(self):
        self.assertEqual(descriptive([10, 1, 4]), {"n": 3, "mean": 5, "median": 4, "minimum": 1, "maximum": 10})
        self.assertIsNone(descriptive([])["mean"])

    def test_preservation_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            original = root / "original.txt"
            original.write_text("original", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": {"original.txt": {"sha256": sha256_file(original)}}}), encoding="utf-8")
            with patch("stage5.preserve.ROOT", root), patch("stage5.preserve.MANIFEST", manifest):
                self.assertEqual(verify_preserved(), 1)
                original.write_text("changed fixture", encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    verify_preserved()


@unittest.skipUnless((OUTPUT / "analysis.json").exists(), "Stage 5 analysis not generated")
class ActualStage5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = json.loads((OUTPUT / "analysis.json").read_text(encoding="utf-8"))

    def test_no_unsupported_p_values_or_population_claims(self):
        data = self.analysis
        self.assertFalse(data["inferential_p_values_produced"])
        for row in data["sentence_effects"] + data["domain_size_adjustment"] + data["h1_contrasts"]:
            self.assertIsNone(row["p_value"])
        self.assertTrue(all(not row["performed"] for row in data["statistical_test_decisions"]))
        self.assertEqual(len(data["sentence_effects"]), 12)
        self.assertEqual(len(data["filter_contrasts"]), 18)

    def test_h1_evidence_is_measured_and_mixed(self):
        for row in self.analysis["h1_contrasts"]:
            if row["target"] == "quality":
                self.assertTrue(row["target_greater_document_reduction"])
                self.assertTrue(row["target_greater_token_reduction"])
            if row["target"] == "dedup" and row["comparator"] == "language":
                self.assertFalse(row["target_greater_document_reduction"])
                self.assertFalse(row["target_greater_token_reduction"])

    def test_diversity_size_adjustment_qualifies_raw_richness_loss(self):
        for row in self.analysis["domain_size_adjustment"]:
            if row["filter"] == "quality":
                self.assertEqual(row["observed_unique_domains"], row["retained_documents"])
                self.assertGreaterEqual(row["observed_unique_domains"], row["expected_unique_uniform_same_k"] - 1e-10)
                self.assertAlmostEqual(row["observed_entropy_bits"], math.log2(row["retained_documents"]))

    def test_c4_uses_paired_document_ids_and_reconciles_totals(self):
        pairs = self.analysis["c4_paired_documents"]
        self.assertEqual(len(pairs), 141)
        self.assertEqual(len({(row["dataset"], row["document_id"]) for row in pairs}), 141)
        for summary in self.analysis["c4_summary"]:
            selected = [row for row in pairs if row["dataset"] == summary["dataset"]]
            self.assertEqual(sum(row["token_change"] for row in selected), summary["token_change"])
            self.assertEqual(summary["saved_tokens"] + summary["token_change"], summary["cleaned_tokens"])
            self.assertTrue(summary["domain_counts_unchanged"])


if __name__ == "__main__":
    unittest.main()
