"""Shared dataset paths, settings and artifact checks."""

import json
from pathlib import Path

from src.filter_assets import ROOT, sha256_file
from src.run_filters import STAGES, load_config
from src.preservation import verify_research_files

EVIDENCE = ROOT / "results/stage35"
RUNS = ROOT / "runs/stage35"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_legacy():
    snapshot = json.loads((EVIDENCE / "legacy_snapshot.json").read_text(encoding="utf-8"))
    return verify_research_files(ROOT, snapshot["files"])


def baseline_path(dataset):
    return ROOT / "data/processed" / ("baseline.jsonl" if str(dataset) == "1" else f"baseline_{dataset}.jsonl")


def run_root(dataset, replay=False):
    return RUNS / ("replay" if replay else "primary") / f"dataset_{dataset}"


def filter_settings(dataset):
    original = load_config(ROOT / "filter_config.json")
    settings = json.loads(json.dumps(original))
    settings["baseline"] = str(baseline_path(dataset).resolve())
    settings["baseline_sha256"] = sha256_file(baseline_path(dataset))
    settings["assets_directory"] = str((ROOT / original["assets_directory"]).resolve())
    assert all(settings[stage] == original[stage] for stage in STAGES)
    assert settings["max_text_characters"] == original["max_text_characters"]
    return settings


def ledger_rows(path):
    with Path(path).open("rb") as stream:
        for line in stream:
            yield json.loads(line)
