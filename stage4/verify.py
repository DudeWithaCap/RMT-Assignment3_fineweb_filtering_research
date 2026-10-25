"""Verify preservation, exported hashes and equality of two actual metric runs."""

import argparse
import json
from pathlib import Path

from src.filter_assets import sha256_file
from .preserve import OUTPUT, verify_preserved


def verify(primary=OUTPUT, replay=OUTPUT / "replay"):
    primary, replay = Path(primary), Path(replay)
    first = json.loads((primary / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((replay / "manifest.json").read_text(encoding="utf-8"))
    if first["inputs"] != second["inputs"] or first["measurement"] != second["measurement"] or first["versions"] != second["versions"]:
        raise ValueError("Runs used different inputs or measurement definitions")
    compared = []
    for name, checksum in second["output_sha256"].items():
        if first["output_sha256"].get(name) != checksum or sha256_file(primary / name) != checksum or sha256_file(replay / name) != checksum:
            raise ValueError(f"Metric replay mismatch: {name}")
        compared.append(name)
    for name, checksum in first["output_sha256"].items():
        if sha256_file(primary / name) != checksum:
            raise ValueError(f"Primary artifact changed: {name}")
    return {"all_metric_files_byte_identical": True, "compared_files": sorted(compared),
            "primary_combinations": first["primary_combinations"], "supplementary_combinations": first["supplementary_combinations"],
            "protected_files_unchanged": verify_preserved()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=OUTPUT)
    parser.add_argument("--replay", type=Path, default=OUTPUT / "replay")
    arguments = parser.parse_args()
    result = verify(arguments.primary, arguments.replay)
    (OUTPUT / "reproducibility.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
