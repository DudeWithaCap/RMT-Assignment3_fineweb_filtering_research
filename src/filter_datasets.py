"""Run independent or sequential filters with dated charts and JSON outputs."""

import argparse
import json
from pathlib import Path

from src.filter_assets import ROOT, sha256_file
from src.run_filters import STAGES, run_stage, load_config
from src.run_results import create_run, inspect_baseline, now, publish, write_json
from src.datasets.common import filter_settings


def run(datasets, stages, output=ROOT / "results", baseline=None, dataset_name=None,
        config_path=ROOT / "filter_config.json", mode="independent", keep_intermediate=False):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "results") or output.is_relative_to(ROOT / "results/archive"):
        raise ValueError("New runs must stay inside results/, outside archive/")
    if not stages or len(set(stages)) != len(stages) or any(stage not in STAGES for stage in stages):
        raise ValueError("Choose one or more distinct valid filters")
    if mode not in {"independent", "sequential"}:
        raise ValueError("Mode must be independent or sequential")
    if mode == "sequential" and tuple(stages) != STAGES:
        raise ValueError("Sequential mode requires all four filters in URL, language, quality, dedup order")
    jobs = []
    if baseline is not None:
        config = load_config(config_path)
        config["baseline"] = str(Path(baseline).resolve())
        config["assets_directory"] = str((ROOT / config["assets_directory"]).resolve())
        count, checksum, crawls = inspect_baseline(Path(config["baseline"]), config["max_text_characters"])
        name = dataset_name or (next(iter(crawls)) if len(crawls) == 1 else None)
        if not name:
            raise ValueError("Use --dataset-name when a custom baseline has no single crawl identifier")
        config.update(expected_documents=count, baseline_sha256=checksum)
        jobs.append((name, config))
    else:
        names = {1: "CC_2025_30", 2: "CC_2025_18", 3: "CC_2025_47"}
        for dataset in datasets:
            config = filter_settings(dataset)
            count, checksum, crawls = inspect_baseline(Path(config["baseline"]), config["max_text_characters"])
            if count != config["expected_documents"] or crawls != {names[dataset]}:
                raise ValueError("Built-in baseline has an unexpected document count or crawl")
            jobs.append((names[dataset], config))
    directories = []
    for name, config in jobs:
        started = now()
        directory = create_run(output, name, started, mode=mode)
        details = directory / "details"
        details.mkdir()
        write_json(details / "filter_config.json", config)
        write_json(directory / "summary.json", {"status": "running", "dataset": name,
                   "filter_mode": mode, "started_at": started.isoformat()})
        try:
            outputs = {}
            current_input = Path(config["baseline"])
            current_count = config["expected_documents"]
            for stage in stages:
                stage_config = dict(config)
                if mode == "sequential":
                    stage_config.update(baseline=str(current_input.resolve()),
                                        baseline_sha256=sha256_file(current_input), expected_documents=current_count)
                run_stage(stage, stage_config, root=details)
                outputs[stage] = {"dataset": details / f"data/processed/filtered_{stage}.jsonl",
                                  "summary": details / f"results/{stage}_summary.json"}
                if mode == "sequential":
                    current_input = outputs[stage]["dataset"]
                    report = json.loads(outputs[stage]["summary"].read_text(encoding="utf-8"))
                    current_count = report["output_count"]
            publish(directory, name, Path(config["baseline"]), config["baseline_sha256"], started, outputs,
                    origin="new_filter_execution", mode=mode, keep_intermediate=keep_intermediate)
        except BaseException as error:
            write_json(directory / "summary.json", {"status": "failed", "dataset": name,
                       "filter_mode": mode, "started_at": started.isoformat(), "failed_at": now().isoformat(),
                       "error": f"{type(error).__name__}: {error}"})
            raise
        directories.append(directory)
        print(f"Saved {mode} results: {directory}", flush=True)
    return directories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--dataset", choices=("1", "2", "3", "all"))
    inputs.add_argument("--input", type=Path, help="Custom baseline JSONL with the existing document schema")
    parser.add_argument("--dataset-name", help="Folder label for a custom baseline, e.g. CC_2026_42")
    parser.add_argument("--config", type=Path, default=ROOT / "filter_config.json", help="Filter settings for a custom baseline")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--mode", choices=("independent", "sequential"), default="independent")
    parser.add_argument("--keep-intermediate", action="store_true", help="Also export intermediate JSON datasets in sequential mode")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results", help="Parent directory; a unique dated subfolder is created")
    args = parser.parse_args()
    if args.dataset_name and not args.input:
        parser.error("--dataset-name requires --input")
    if args.mode == "sequential" and args.stage != "all":
        parser.error("Sequential mode requires --stage all")
    run((1, 2, 3) if args.dataset in (None, "all") else (int(args.dataset),),
        STAGES if args.stage == "all" else (args.stage,), args.output_dir,
        baseline=args.input, dataset_name=args.dataset_name, config_path=args.config,
        mode=args.mode, keep_intermediate=args.keep_intermediate)
