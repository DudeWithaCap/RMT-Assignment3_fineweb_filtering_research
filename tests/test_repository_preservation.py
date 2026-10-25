from pathlib import Path
import tempfile
import unittest

from src.filter_assets import sha256_file
from src.preservation import verify_research_files


class RepositoryPreservationTests(unittest.TestCase):
    def test_editable_readme_does_not_invalidate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            readme = root / "README.md"
            readme.write_text("old")
            records = {"README.md": {"sha256": sha256_file(readme)}}
            readme.write_text("new")
            self.assertEqual(verify_research_files(root, records), 0)

    def test_missing_and_modified_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/metrics.json"
            output.parent.mkdir()
            output.write_text("original")
            records = {"results/metrics.json": {"sha256": sha256_file(output)}}
            self.assertEqual(verify_research_files(root, records), 1)
            output.write_text("changed")
            with self.assertRaises(RuntimeError):
                verify_research_files(root, records)
            output.unlink()
            with self.assertRaises(RuntimeError):
                verify_research_files(root, records)

    def test_local_datasets_optional_in_clone_but_checked_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "data/processed/baseline.jsonl"
            local.parent.mkdir(parents=True)
            local.write_text("original")
            records = {"data/processed/baseline.jsonl": {"sha256": sha256_file(local)}}
            local.unlink()
            self.assertEqual(verify_research_files(root, records), 0)
            local.write_text("changed")
            with self.assertRaises(RuntimeError):
                verify_research_files(root, records)
