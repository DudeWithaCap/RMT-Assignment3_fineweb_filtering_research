"""Run independent filters with optional replay verification."""

import argparse
import hashlib
import json

from src.filter_assets import sha256_file
from src.run_filters import STAGES, baseline_rows, run_stage
from .common import EVIDENCE, baseline_path, check_legacy, filter_settings, ledger_rows, run_root, write_json


def validate_stage(stage, config, root):
    baseline = baseline_path_from_config(config)
    originals = {row["document_id"]: (line, row) for line, row in baseline_rows(baseline, config["max_text_characters"])}
    summary = json.loads((root / f"results/{stage}_summary.json").read_text(encoding="utf-8"))
    output = root / f"data/processed/filtered_{stage}.jsonl"
    decisions_file = root / f"results/{stage}_decisions.jsonl"
    removed_file = root / f"results/{stage}_removed.jsonl"
    assert summary["status"] == "complete"
    assert summary["config"] == config
    assert summary["baseline_sha256"] == config["baseline_sha256"] == summary["baseline_sha256_after"] == sha256_file(baseline)
    assert len(originals) == summary["input_count"] == config["expected_documents"]
    ids_digest = hashlib.sha256()
    for identity in originals:
        ids_digest.update((identity + "\n").encode())
    assert ids_digest.hexdigest() == summary["input_ids_sha256"]
    decisions = list(ledger_rows(decisions_file))
    assert [record["document_id"] for record in decisions] == list(originals)
    for decision in decisions:
        original = originals[decision["document_id"]][1]
        assert all(decision[key] == original[key] for key in ("url", "domain", "source"))
    kept = []
    with output.open("rb") as stream:
        for line in stream:
            document = json.loads(line)
            assert line == originals[document["document_id"]][0], "Output document changed"
            kept.append(document["document_id"])
    assert kept == [record["document_id"] for record in decisions if record["kept"]]
    removed = list(ledger_rows(removed_file))
    assert removed == [record for record in decisions if not record["kept"]]
    assert len(kept) == summary["output_count"]
    assert len(removed) == summary["removed_count"]
    assert sha256_file(output) == summary["output_sha256"]
    assert sha256_file(decisions_file) == summary["decisions_sha256"]
    return summary


def baseline_path_from_config(config):
    from pathlib import Path
    return Path(config["baseline"])


def run_dataset(dataset, replay=False):
    check_legacy()
    config = filter_settings(dataset)
    root = run_root(dataset, replay)
    path = root / "filter_config.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == config, "Saved run configuration changed"
    else:
        write_json(path, config)
    for stage in STAGES:
        summary_path = root / f"results/{stage}_summary.json"
        if not summary_path.exists():
            run_stage(stage, config, root=root)
        validate_stage(stage, config, root)
    report = {"dataset": int(dataset), "baseline_sha256": sha256_file(baseline_path(dataset)),
              "all_stages_verified": True, "replay": replay}
    if replay:
        primary = run_root(dataset)
        for stage in STAGES:
            first = validate_stage(stage, config, primary)
            second = validate_stage(stage, config, root)
            for key in ("baseline_sha256", "input_ids_sha256", "output_sha256", "decisions_sha256", "output_count", "removed_count"):
                assert first[key] == second[key], f"Non-reproducible {stage}: {key}"
        report["actual_second_filter_run_hashes_match"] = True
    check_legacy()
    write_json(EVIDENCE / f"dataset_{dataset}" / ("filter_replay.json" if replay else "filter_verification.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["2", "3", "all"], default="all")
    parser.add_argument("--verify-replay", action="store_true", help="Run each new dataset a second time in a separate root and compare hashes")
    arguments = parser.parse_args()
    for dataset in ("2", "3") if arguments.dataset == "all" else (arguments.dataset,):
        print(f"Dataset {dataset}: primary independent filters", flush=True)
        run_dataset(dataset)
        if arguments.verify_replay:
            print(f"Dataset {dataset}: second run for reproducibility", flush=True)
            run_dataset(dataset, replay=True)
