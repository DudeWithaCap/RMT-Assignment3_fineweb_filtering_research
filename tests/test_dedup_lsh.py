import json
from pathlib import Path
import random
import tempfile
import unittest

from src.dedup_filter import DedupFilter
from src.dedup_lsh import LSHDedupFilter, bucket_pairs
from src.filter_assets import ROOT
from src.run_filters import load_config, run_stage
from test_run_results import fixture

import numpy as np


class WhitespaceTokenizer:
    asset_record = {"fixture": True}

    def word_tokenize(self, text):
        return text.split()


CONFIG = {"ngram_size": 5, "similarity_threshold": 0.8, "max_documents": 100000, "method": "minhash_lsh",
          "num_bands": 20, "rows_per_band": 5, "max_bucket_pairs": 64, "seed": 42, "workers": 1}


def word(number):
    # Letters only: normalization maps every digit to 0, which would merge numbered words.
    letters = ""
    for _ in range(3):
        number, remainder = divmod(number, 26)
        letters += chr(97 + remainder)
    return letters


def corpus(seed=0, bases=40):
    rng = random.Random(seed)
    vocabulary = [word(i) for i in range(400)]
    texts = []
    for _ in range(bases):
        base = [rng.choice(vocabulary) for _ in range(rng.randint(30, 120))]
        texts.append(" ".join(base))
        for edits in rng.sample([0, 1, 2, 4, 8, 20], 3):
            variant = list(base)
            for _ in range(edits):
                variant[rng.randrange(len(variant))] = rng.choice(vocabulary)
            texts.append((" ".join(variant)).upper() if edits == 0 else " ".join(variant))
    texts.append(texts[-1] + " 42")  # normalizes to the same text as the previous document
    texts += ["", "too short", "w1 w2 w3 w4", ""]
    rng.shuffle(texts)
    return [{"document_id": f"d{i}", "text": text} for i, text in enumerate(texts)]


def exact_decisions(documents, config=CONFIG):
    with tempfile.TemporaryDirectory() as directory:
        stage = DedupFilter(config, WhitespaceTokenizer(), Path(directory))
        for document in documents:
            stage.add(document)
        decisions = stage.decisions()
        stage.close()
    return decisions


def lsh_decisions(documents, config=CONFIG, workers=1):
    stage = LSHDedupFilter(config, WhitespaceTokenizer, WhitespaceTokenizer.asset_record, workers=workers)
    decisions = stage.decide_stream(lambda: ((None, d) for d in documents))
    return decisions, stage


def kept(decisions):
    return {identity for identity, (keep, _, _) in decisions.items() if keep}


class LSHDedupTests(unittest.TestCase):
    def test_matches_exact_filter_on_near_duplicate_corpus(self):
        for seed in (0, 1, 2):
            documents = corpus(seed)
            exact = exact_decisions(documents)
            lsh, stage = lsh_decisions(documents)
            self.assertEqual(kept(lsh), kept(exact), f"seed {seed}")
            self.assertEqual({i: d[1] for i, d in lsh.items()}, {i: d[1] for i, d in exact.items()})
            self.assertGreater(len(documents) - len(kept(exact)), 30)  # the corpus really has duplicates
            self.assertGreater(stage.statistics["candidate_pairs"], 0)

    def test_signatures_agree_at_the_jaccard_rate(self):
        import src.dedup_lsh as lsh
        rng = np.random.default_rng(0)
        shared = rng.integers(0, 2 ** 63, 900, dtype=np.uint64)
        first = np.r_[shared, rng.integers(0, 2 ** 63, 50, dtype=np.uint64)]
        second = np.r_[shared, rng.integers(0, 2 ** 63, 50, dtype=np.uint64)]  # J = 0.9
        seeds = np.random.default_rng(1).integers(0, 2 ** 63, 2000, dtype=np.uint64)
        agreement = (lsh.mix(first, seeds).min(0) == lsh.mix(second, seeds).min(0)).mean()
        self.assertAlmostEqual(agreement, .9, delta=.03)

    def test_worker_count_does_not_change_decisions(self):
        documents = corpus(5)
        single, _ = lsh_decisions(documents, workers=1)
        multiple, stage = lsh_decisions(documents, workers=2)
        self.assertEqual(stage.workers, 2)
        self.assertEqual(single, multiple)

    def test_large_buckets_stay_connected(self):
        documents = corpus(3)
        small = dict(CONFIG, max_bucket_pairs=2)
        self.assertEqual(kept(lsh_decisions(documents, small)[0]), kept(exact_decisions(documents)))
        low, high = bucket_pairs(np.array([3, 5, 9, 12]), 2)
        self.assertEqual(sorted(zip(low.tolist(), high.tolist())), [(3, 5), (3, 9), (3, 12), (5, 9), (9, 12)])

    def test_transitive_threshold_semantics_match(self):
        words = [word(i) for i in range(60)]
        a = " ".join(words)
        b = " ".join(words[:57] + ["x", "y", "z"])            # similar to a
        c = " ".join(words[:54] + ["x", "y", "z", "p", "q", "r"])  # similar to b, less to a
        documents = [{"document_id": name, "text": text} for name, text in (("a", a), ("b", b), ("c", c))]
        exact = exact_decisions(documents)
        lsh, _ = lsh_decisions(documents)
        self.assertEqual(kept(lsh), kept(exact))
        self.assertEqual(lsh["b"][2]["representative_id"], "a")

    def test_runner_uses_streaming_hook_and_config_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            rows = [fixture(i) for i in range(4)]
            for i, row in enumerate(rows):
                row["text"] = " ".join(word(i * 50 + j) for j in range(40))
            rows[2]["text"] = rows[1]["text"].upper() + "!!"  # same normalized text
            for row in rows:
                row["character_count"] = row["original_character_count"] = len(row["text"])
            baseline = root / "baseline.jsonl"
            baseline.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            config = load_config(ROOT / "filter_config_1m.json")
            config.update(baseline=str(baseline), baseline_sha256=None, expected_documents=4)
            config["dedup"]["workers"] = 1
            factory = lambda *a: LSHDedupFilter(config["dedup"], WhitespaceTokenizer, {"fixture": True})
            report = run_stage("dedup", config, root=root, factory=factory)
            self.assertEqual(report["removed_count"], 1)
            self.assertEqual(report["removal_reasons"], {"dedup_normalized_exact": 1})
            self.assertEqual(report["assets"]["minhash_lsh"]["documents"], 4)
        broken = load_config(ROOT / "filter_config.json")
        broken["dedup"]["max_documents"] = 6000
        path = Path(tempfile.mkdtemp(dir=ROOT / "tests")) / "config.json"
        path.write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, "minhash_lsh"):
            load_config(path)
        path.unlink()
        path.parent.rmdir()


if __name__ == "__main__":
    unittest.main()
