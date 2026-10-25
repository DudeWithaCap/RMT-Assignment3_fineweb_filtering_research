"""Create an initial preservation manifest once, or verify the existing one."""

from src.filter_assets import ROOT, sha256_file
from .common import EVIDENCE, baseline_path, check_legacy, write_json


def preserve():
    snapshot = EVIDENCE / "legacy_snapshot.json"
    if snapshot.exists():
        print(f"Verified {check_legacy()} original files unchanged")
        return
    if any(baseline_path(dataset).exists() for dataset in (2, 3)):
        raise RuntimeError("New baselines exist but the initial preservation manifest is missing; restore it rather than resetting history")
    excluded = {".venv", "venv", "__pycache__", ".git", "tmp", "runs"}
    files = {}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in excluded for part in relative.parts) or path.is_relative_to(EVIDENCE):
            continue
        files[relative.as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(snapshot, {"files": files})
    print(f"Protected {len(files)} existing files")


if __name__ == "__main__":
    preserve()
