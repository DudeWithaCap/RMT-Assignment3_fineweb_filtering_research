"""Protect historical research artifacts during analysis."""

import json
from src.filter_assets import ROOT, sha256_file
from src.preservation import verify_research_files

OUTPUT = ROOT / "results/stage5"
MANIFEST = OUTPUT / "preservation_manifest.json"


def verify_preserved():
    original = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    return verify_research_files(ROOT, original)


def initialize():
    if MANIFEST.exists():
        print(f"Verified {verify_preserved()} prior files unchanged")
        return
    if (OUTPUT / "analysis.json").exists() or (ROOT / "STAGE5_REPORT.md").exists():
        raise RuntimeError("Restore the initial preservation manifest; do not reset history")
    excluded = {".venv", "venv", "__pycache__", ".git", "tmp", ".pytest_cache"}
    files = {}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_file() and not path.is_relative_to(OUTPUT) and not any(part in excluded for part in relative.parts):
            files[relative.as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
    print(f"Protected {len(files)} files")


if __name__ == "__main__":
    initialize()
