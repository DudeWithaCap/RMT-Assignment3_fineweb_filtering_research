"""Run unchanged independent filters into a new, portable experiment directory."""

import argparse
from pathlib import Path

from src.filter_assets import ROOT
from src.run_filters import STAGES, run_stage
from stage35.common import filter_settings


def run(datasets, stages, output):
    output = Path(output).resolve()
    settings = {dataset: filter_settings(dataset) for dataset in datasets}
    if output.exists():
        raise FileExistsError("Choose a new --output-dir to preserve existing results")
    if not output.is_relative_to(ROOT):
        raise ValueError("Output directory must stay inside this project")
    for dataset in datasets:
        for stage in stages:
            result = run_stage(stage, settings[dataset], root=output / f"dataset_{dataset}")
            print(f"Dataset {dataset}/{stage}: {result['input_count']} -> {result['output_count']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/reproduction")
    args = parser.parse_args()
    run((1, 2, 3) if args.dataset == "all" else (int(args.dataset),),
        STAGES if args.stage == "all" else (args.stage,), args.output_dir)
