"""Check numeric and figure replay against preserved inputs."""

import argparse
import json
from pathlib import Path

from src.filter_assets import sha256_file
from stage4.collect import write_json
from .preserve import OUTPUT, verify_preserved


def verify(primary=OUTPUT, replay=OUTPUT / "replay"):
    primary, replay = Path(primary), Path(replay)
    first = json.loads((primary / "analysis_manifest.json").read_text(encoding="utf-8"))
    second = json.loads((replay / "analysis_manifest.json").read_text(encoding="utf-8"))
    if first["inputs_sha256"] != second["inputs_sha256"] or first["output_sha256"] != second["output_sha256"]:
        raise ValueError("Statistical replay uses different inputs/results")
    for name, digest in first["output_sha256"].items():
        if sha256_file(primary / name) != digest or sha256_file(replay / name) != digest:
            raise ValueError(f"Numeric replay mismatch: {name}")
    figures = json.loads((primary / "figures/figures_manifest.json").read_text(encoding="utf-8"))
    replay_figures = json.loads((replay / "figures/figures_manifest.json").read_text(encoding="utf-8"))
    if figures != replay_figures:
        raise ValueError("Figure replay manifests differ")
    for record in figures:
        for extension in ("png", "svg"):
            name = record["name"] + "." + extension
            checksum = record[extension + "_sha256"]
            if sha256_file(primary / "figures" / name) != checksum or sha256_file(replay / "figures" / name) != checksum:
                raise ValueError(f"Figure replay mismatch: {name}")
    return {"numeric_files_byte_identical": len(first["output_sha256"]), "figures": len(figures),
            "png_svg_files_byte_identical": len(figures) * 2, "prior_files_unchanged": verify_preserved()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=OUTPUT / "replay")
    args = parser.parse_args()
    result = verify(replay=args.replay)
    write_json(OUTPUT / "reproducibility.json", result)
    print(json.dumps(result, indent=2))
