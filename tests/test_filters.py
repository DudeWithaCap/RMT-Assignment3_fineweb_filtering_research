from copy import deepcopy
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from src.dedup_filter import DedupFilter
from src.filter_assets import ROOT, checked_asset, sha256_file
from src.filter_text import simplify_text
from src.language_filter import LanguageFilter
from src.quality_filter import QualityFilter
from src.run_filters import STAGES, load_config, run_stage
from src.url_filter import URLFilter

ARTICLE = (
    "The university research team designed a careful study of public internet archives and their changing composition. "
    "Independent observations helped investigators understand how different collection procedures influence the available material. "
    "Several experienced analysts compared records with original sources to ensure that each measurement remained reliable. "
    "Participants discussed practical questions about transparent documentation and responsible management of historical information. "
    "Their detailed report explained important limitations while describing useful methods for future scientific investigations. "
    "Readers can examine the evidence and assess whether these conclusions reflect the particular collection under consideration."
)


class FixtureTokenizer:
    """Deliberately simple offline test double, never used by production filters."""
    asset_record = {"fixture": True}

    def word_tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text)

    def sent_tokenize(self, text):
        return [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


class FixtureModel:
    def __init__(self, language="en", score=0.95):
        self.language, self.score = language, score
        self.calls = []

    def predict(self, text, k=1):
        self.calls.append(text)
        return ["__label__" + self.language], [self.score]


def document(index, text=ARTICLE, url="https://example.org/article"):
    return {"document_id": f"fixture-{index}", "text": text, "url": url,
            "domain": "example.org", "character_count": len(text),
            "original_character_count": len(text), "truncated": False,
            "source": {"wet_path": "fixture.warc.wet.gz", "warc_record_id": f"urn:fixture:{index}",
                       "record_index": index, "warc_date": "2024-01-01T00:00:00Z"}}


def miniature_assets(root):
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(root / "url_filterblacklists.tar.gz", "w:gz") as archive:
        for name, text in (("adult/domains", "blocked.com\n"), ("adult/urls", "https://exact.example/path\n")):
            data = text.encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    for name, text in (("banned_words.txt", "forbidden\n"), ("banned_subwords.txt", "badsubstring\n"),
                       ("soft_banned_words.txt", "softone\nsofttwo\n")):
        (root / name).write_text(text, encoding="utf-8")


def fixture_asset(directory, name):
    path = Path(directory) / name
    return path, {"sha256": sha256_file(path), "fixture": True}


class FilterUnitTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "filter_config.json")
        self.tokenizer = FixtureTokenizer()

    def test_url_all_six_rules_and_allow(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            miniature_assets(root)
            config = deepcopy(self.config["url"])
            config["extra_domains"] = ["nested.allowed.org"]
            with patch("src.url_filter.checked_asset", side_effect=fixture_asset):
                stage = URLFilter(config, root, root)
            try:
                for url, reason in (
                        ("https://blocked.com/page", "domain"),
                        ("https://nested.allowed.org/page", "subdomain"),
                        ("https://exact.example/path", "url"),
                        ("https://allowed.org/forbidden", "hard_blacklisted"),
                        ("https://allowed.org/softone/softtwo", "soft_blacklisted"),
                        ("https://allowed.org/bad-substring", "blacklisted_subword")):
                    with self.subTest(url=url):
                        kept, actual, details = stage.evaluate({"url": url})
                        self.assertFalse(kept)
                        self.assertEqual(actual, "url_" + reason)
                        self.assertTrue(details["matched"])
                self.assertTrue(stage.evaluate({"url": "https://allowed.org/article"})[0])
                self.assertTrue(stage.evaluate({"url": "https://allowed.org/softone/softone"})[0])
            finally:
                stage.close()

    def test_language_prediction_confidence_boundaries_and_no_mutation(self):
        for language, score, reason in (("en", 0.9, None), ("fr", 0.99, "language_non_english"),
                                        ("en", 0.65, "language_low_confidence"),
                                        ("en", 0.64, "language_low_confidence")):
            model = FixtureModel(language, score)
            stage = LanguageFilter(self.config["language"], None, model=model)
            original = document(0, "the\ntext")
            copied = deepcopy(original)
            kept, actual, _ = stage.evaluate(original)
            self.assertEqual(actual, reason)
            self.assertEqual(kept, reason is None)
            self.assertEqual(original, copied)
            self.assertEqual(model.calls, ["thetext"])
        self.assertEqual(stage.evaluate(document(0, "\n  "))[1], "language_empty_text")

    def test_quality_accepts_prose_rejects_short_and_repetition(self):
        stage = QualityFilter(self.config["quality"], self.tokenizer)
        good = document(0)
        original = deepcopy(good)
        kept, reason, _ = stage.evaluate(good)
        self.assertTrue(kept, reason)
        self.assertEqual(good, original)
        short_config = deepcopy(self.config["quality"])
        short_config["top_ngram_thresholds"] = {}
        short_config["duplicate_ngram_thresholds"] = {}
        self.assertEqual(QualityFilter(short_config, self.tokenizer).evaluate(
            document(1, "A very short document."))[1], "quality_short_doc")
        self.assertFalse(stage.evaluate(document(2, "Repeated example line.\n" * 50))[0])
        self.assertEqual(stage.evaluate(document(3, ""))[1], "quality_empty_text")

    def test_quality_threshold_changes_and_c4_rules(self):
        config = deepcopy(self.config["quality"])
        config["min_doc_words"] = 1000
        self.assertEqual(QualityFilter(config, self.tokenizer).evaluate(document(0))[1], "quality_short_doc")
        for text, reason in ((ARTICLE + " Lorem ipsum appears in this extra sentence.", "quality_c4_lorem_ipsum"),
                             (ARTICLE + " The unexpected { symbol occurs within this sentence.", "quality_c4_curly_bracket")):
            self.assertEqual(QualityFilter(self.config["quality"], self.tokenizer).evaluate(document(0, text))[1], reason)
        text = ARTICLE + "\nThis website uses cookies to store important user preferences."
        good = document(0, text)
        kept, reason, details = QualityFilter(self.config["quality"], self.tokenizer).evaluate(good)
        self.assertTrue(kept, reason)
        self.assertEqual(details["c4_removed_lines"], {"policy": 1})
        self.assertEqual(good["text"], text)

    def test_fineweb_custom_thresholds(self):
        for key, value, expected in (("line_punct_threshold", 1, "quality_line_punct_ratio"),
                                      ("short_line_length", 10000, "quality_short_line_ratio"),
                                      ("char_duplicates_ratio", 0, "quality_char_dup_ratio")):
            config = deepcopy(self.config["quality"])
            config[key] = value
            self.assertEqual(QualityFilter(config, self.tokenizer).evaluate(document(0))[1], expected)

    def test_dedup_exact_near_short_empty_and_distinct(self):
        texts = [ARTICLE, ARTICLE, ARTICLE + " A small additional statement follows.",
                 "Completely separate vocabulary concerns stellar astronomy.", "tiny", "tiny", "", ""]
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            stage = DedupFilter(self.config["dedup"], self.tokenizer, Path(directory))
            try:
                for index, text in enumerate(texts):
                    stage.add(document(index, text))
                decisions = stage.decisions()
                self.assertEqual([decisions[f"fixture-{i}"][0] for i in range(len(texts))],
                                 [True, False, False, True, True, False, True, False])
                self.assertEqual(decisions["fixture-1"][2]["representative_id"], "fixture-0")
                self.assertGreaterEqual(decisions["fixture-2"][2]["similarity"], 0.8)
            finally:
                stage.close()

    def test_dedup_transitive_components_and_threshold(self):
        # Word unigram sets: A~B=2/3, B~C=2/3, A~C=1/3.
        config = {"ngram_size": 1, "similarity_threshold": 0.6, "max_documents": 20}
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            stage = DedupFilter(config, self.tokenizer, Path(directory))
            try:
                for i, text in enumerate(("apple banana", "apple banana cherry", "banana cherry")):
                    stage.add(document(i, text))
                result = stage.decisions()
                self.assertTrue(result["fixture-0"][0])
                self.assertFalse(result["fixture-2"][0])
                self.assertTrue(result["fixture-2"][2]["transitive_cluster"])
            finally:
                stage.close()

    def test_normalization(self):
        self.assertEqual(simplify_text(" CAFÉ 123!\n"), "cafe 0")

    def test_invalid_configuration(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "config.json"
            for key, value in (("english_confidence_threshold", -0.1), ("english_confidence_threshold", float("nan"))):
                config = deepcopy(self.config)
                config["language"][key] = value
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)


class PipelineTests(unittest.TestCase):
    def test_independent_inputs_preservation_and_reproducibility(self):
        config = load_config(ROOT / "filter_config.json")
        rows = [document(0), document(1), document(2, "Very short text."),
                document(3, ARTICLE + " A new ending follows.")]
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            baseline = root / config["baseline"]
            baseline.parent.mkdir(parents=True)
            baseline.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            before = baseline.read_bytes()
            config["baseline_sha256"] = sha256_file(baseline)
            config["expected_documents"] = len(rows)
            miniature_assets(root / config["assets_directory"])

            def factory(stage, settings, assets, workspace):
                if stage == "url":
                    with patch("src.url_filter.checked_asset", side_effect=fixture_asset):
                        return URLFilter(settings[stage], assets, workspace)
                if stage == "language":
                    return LanguageFilter(settings[stage], assets, model=FixtureModel())
                if stage == "quality":
                    return QualityFilter(settings[stage], FixtureTokenizer())
                return DedupFilter(settings[stage], FixtureTokenizer(), workspace)

            reports = [run_stage(stage, config, root=root, factory=factory) for stage in STAGES]
            self.assertEqual({r["baseline_sha256"] for r in reports}, {config["baseline_sha256"]})
            self.assertEqual(len({r["input_ids_sha256"] for r in reports}), 1)
            for stage, report in zip(STAGES, reports):
                with self.subTest(stage=stage):
                    self.assertEqual(report["input_count"], len(rows))
                    self.assertEqual(report["output_count"] + report["removed_count"], len(rows))
                    self.assertEqual(baseline.read_bytes(), before)
                    output = root / f"data/processed/filtered_{stage}.jsonl"
                    decisions = root / f"results/{stage}_decisions.jsonl"
                    first_bytes, first_decisions = output.read_bytes(), decisions.read_bytes()
                    for line in first_bytes.splitlines():
                        row = json.loads(line)
                        self.assertEqual(row, rows[int(row["document_id"].split("-")[1])])
                    self.assertEqual(len(first_decisions.splitlines()), len(rows))
                    run_stage(stage, config, root=root, factory=factory, overwrite=True)
                    self.assertEqual(first_bytes, output.read_bytes())
                    self.assertEqual(first_decisions, decisions.read_bytes())

    def test_rejects_changed_baseline_without_publishing(self):
        config = load_config(ROOT / "filter_config.json")
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            baseline = root / config["baseline"]
            baseline.parent.mkdir(parents=True)
            baseline.write_text(json.dumps(document(0)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_stage("url", config, root=root)
            self.assertFalse((baseline.parent / "filtered_url.jsonl").exists())
            report = json.loads((root / "results/url_failure.json").read_text())
            self.assertEqual(report["status"], "failed")


@unittest.skipUnless((ROOT / "data/assets/assets_manifest.json").exists(), "Optional local-asset smoke tests: run src.filter_assets first")
class LocalAssetTests(unittest.TestCase):
    def test_pretrained_fasttext_on_small_real_language_examples(self):
        config = load_config(ROOT / "filter_config.json")
        stage = LanguageFilter(config["language"], ROOT / config["assets_directory"])
        self.assertTrue(stage.evaluate(document(0, ARTICLE))[0])
        self.assertFalse(stage.evaluate(document(1, "Ceci est un texte français qui explique les méthodes de recherche et les résultats scientifiques."))[0])

    def test_nltk_tokenizer_and_asset_integrity(self):
        from src.filter_tokenizer import EnglishTokenizer
        directory = ROOT / "data/assets"
        tokenizer = EnglishTokenizer(directory)
        self.assertEqual(tokenizer.sent_tokenize("First sentence. Second sentence!"), ["First sentence.", "Second sentence!"])
        self.assertIn("research", tokenizer.word_tokenize(ARTICLE))
        self.assertTrue(QualityFilter(load_config(ROOT / "filter_config.json")["quality"], tokenizer).evaluate(document(0))[0])
        path, record = checked_asset(directory, "lid.176.ftz")
        self.assertEqual(sha256_file(path), record["sha256"])


if __name__ == "__main__":
    unittest.main()
