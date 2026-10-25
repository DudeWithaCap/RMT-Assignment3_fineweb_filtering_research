from copy import deepcopy
import json
import unittest
from pathlib import Path

from src.filter_assets import ROOT
from src.filter_text import PUNCTUATION_SET
from src.filter_tokenizer import EnglishTokenizer
from src.quality_filter import QualityFilter
from src.run_filters import load_config
from validation.audit_stage3 import review_sample
from validation.reference_quality import build_reference, reference_decision

TEXT = (
    "The university research team designed a careful study of public internet archives and their changing composition. "
    "Independent observations helped investigators understand how different collection procedures influence the available material. "
    "Several experienced analysts compared records with original sources to ensure that each measurement remained reliable. "
    "Participants discussed practical questions about transparent documentation and responsible management of historical information. "
    "Their detailed report explained important limitations while describing useful methods for future scientific investigations. "
    "Readers can examine the evidence and assess whether these conclusions reflect the particular collection under consideration."
)


class ReviewSamplingTests(unittest.TestCase):
    def test_review_selection_is_balanced_and_order_independent(self):
        rows = []
        for kind in ("kept", "low", "non"):
            for index in range(15):
                rows.append({"document_id": f"fixture-{kind}-{index}", "kept": kind == "kept",
                             "reason": None if kind == "kept" else "language_low_confidence" if kind == "low" else "language_non_english"})
        before = deepcopy(rows)
        sample = review_sample(rows)
        self.assertEqual(len(sample), 30)
        self.assertEqual(sample, review_sample(list(reversed(rows))))
        self.assertEqual(rows, before)
        for stratum in ("kept_english", "low_confidence_english", "predicted_non_english"):
            self.assertEqual(sum(row["stratum"] == stratum for row in sample), 10)

    def test_frozen_review_has_stable_document_ids_and_explicit_uncertainty(self):
        review = json.loads((ROOT / "validation/language_review.json").read_text(encoding="utf-8"))
        identities = [row["document_id"] for row in review["judgments"]]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(len(identities), 30)
        self.assertTrue(all(len(identity) == 64 for identity in identities))
        self.assertIn("not an independent human gold standard", review["reviewer"])
        self.assertTrue(any(row["label"] == "indeterminate" for row in review["judgments"]))


REFERENCE_DIRECTORY = ROOT / "data/assets/validation_reference"


@unittest.skipUnless(REFERENCE_DIRECTORY.exists(), "Validation reference sources not cached; run validation.audit_stage3 --fetch-reference")
class UpstreamParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "filter_config.json")
        cls.tokenizer = EnglishTokenizer(ROOT / cls.config["assets_directory"])
        cls.reference, cls.namespace = build_reference(REFERENCE_DIRECTORY, cls.tokenizer)
        cls.current = QualityFilter(cls.config["quality"], cls.tokenizer)

    def test_reference_defaults_and_punctuation_match(self):
        self.assertEqual(self.namespace["PUNCTUATION_SET"], PUNCTUATION_SET)
        repetition, quality, c4, custom = self.reference
        self.assertEqual(quality.min_doc_words, self.config["quality"]["min_doc_words"])
        self.assertEqual(quality.max_non_alpha_words_ratio, self.config["quality"]["min_alpha_word_ratio"])
        self.assertEqual(custom.char_duplicates_ratio, self.config["quality"]["char_duplicates_ratio"])
        self.assertFalse(c4.filter_no_terminal_punct)
        self.assertEqual(repetition.dup_line_frac, self.config["quality"]["dup_line_frac"])

    def test_default_reference_decisions_match_small_fixtures(self):
        for text in (TEXT, "A short document.", "Repeated line.\n" * 30,
                     TEXT + " Lorem ipsum appears in this additional sentence.",
                     TEXT + " An opening { bracket appears in this sentence."):
            with self.subTest(prefix=text[:30]):
                expected = reference_decision(self.reference, text)[0]
                document = {"text": text}
                before = deepcopy(document)
                actual = self.current.evaluate(document)[0]
                self.assertEqual(actual, expected)
                self.assertEqual(document, before)

    def test_original_c4_changes_text_while_current_retention_keeps_original(self):
        from types import SimpleNamespace

        text = TEXT + "\nThis website uses cookies for user preferences.\n"
        reference_document = SimpleNamespace(text=text)
        self.assertTrue(self.reference[2].filter(reference_document))
        self.assertNotEqual(reference_document.text, text)
        document = {"text": text}
        self.assertTrue(self.current.evaluate(document)[0])
        self.assertEqual(document["text"], text)


if __name__ == "__main__":
    unittest.main()
