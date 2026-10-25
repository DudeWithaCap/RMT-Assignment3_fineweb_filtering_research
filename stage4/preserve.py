"""Protect historical research artifacts during metric collection."""

import json

from src.filter_assets import ROOT, sha256_file
from src.preservation import verify_research_files

OUTPUT = ROOT / "results/stage4"
MANIFEST = OUTPUT / "preservation_manifest.json"


def verify_preserved():
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return verify_research_files(ROOT, record["files"])


def initialize():
    if MANIFEST.exists():
        print(f"Verified {verify_preserved()} existing files unchanged")
        return
    if (OUTPUT / "metrics.json").exists() or (ROOT / "STAGE4_REPORT.md").exists():
        raise RuntimeError("Restore the original Stage 4 preservation manifest; do not reset preservation history")
    excluded = {".venv", "venv", "__pycache__", ".git", "tmp", ".pytest_cache"}
    paths = [path for path in ROOT.rglob("*") if path.is_file() and not path.is_relative_to(OUTPUT)
             and not any(part in excluded for part in path.relative_to(ROOT).parts)]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"files": {path.relative_to(ROOT).as_posix():
                        {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths}}, indent=2) + "\n", encoding="utf-8")
    print(f"Protected {len(paths)} files")


if __name__ == "__main__":
    initialize()
