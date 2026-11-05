"""Interactive terminal menu for dataset filtering."""

from pathlib import Path

from .filter_assets import ROOT
from .filter_datasets import run
from .run_filters import STAGES


def choose(prompt, options, read, write):
    while True:
        write("\n" + prompt)
        for key, label in options.items():
            write(f"  {key} - {label}")
        answer = read("Your choice: ").strip()
        if answer in options:
            return answer
        write("Invalid choice. Enter one of the numbers shown above.")


def main(input_fn=None, output_fn=None, runner=None):
    read, write, execute = input_fn or input, output_fn or print, runner or run
    write("FineWeb Filtering Research")
    write("Every run saves a separate dated results folder.")
    try:
        while True:
            selected_mode = choose("Choose filtering mode", {
                "1": "Independent (test each filter separately on the same dataset)",
                "2": "Sequential (run one dataset through all four filters in order)",
                "0": "Exit"}, read, write)
            if selected_mode == "0":
                return 0
            mode = "independent" if selected_mode == "1" else "sequential"
            selected_dataset = choose("Choose input dataset", {
                "1": "CC-MAIN-2025-30", "2": "CC-MAIN-2025-18", "3": "CC-MAIN-2025-47",
                "4": "All three datasets (one at a time)", "5": "Custom baseline JSONL", "0": "Back"}, read, write)
            if selected_dataset == "0":
                continue
            baseline, name = None, None
            if selected_dataset == "5":
                while True:
                    value = read("Baseline JSONL path (0 to go back): ").strip().strip('"').strip("'")
                    if value == "0":
                        break
                    baseline = Path(value).expanduser()
                    if not baseline.is_absolute():
                        baseline = ROOT / baseline
                    if value and baseline.is_file():
                        break
                    write("File not found. Enter an existing baseline JSONL path.")
                if value == "0":
                    continue
                name = read("Dataset name (Enter to infer from crawl metadata): ").strip() or None
                datasets = ()
            else:
                datasets = (1, 2, 3) if selected_dataset == "4" else (int(selected_dataset),)
            write(f"\nStarting {mode} filtering: URL, Language, Quality, Deduplication")
            if mode == "independent":
                write("Each filter starts from the same input dataset; one result file is saved per filter.")
            else:
                write("The selected dataset passes through all four filters; one final cleaned JSON is saved.")
            try:
                directories = execute(datasets, STAGES, ROOT / "results", baseline=baseline,
                                      dataset_name=name, mode=mode)
            except Exception as error:
                write(f"Run failed: {error}")
                write("Check the input file and local assets. Download assets with: python -m src.filter_assets")
                continue
            for directory in directories:
                write(f"\nCompleted: {directory}")
                write(f"Chart: {directory / 'retention.png'}")
                write(f"Cleaned JSON: {directory / 'cleaned.json' if mode == 'sequential' else directory / 'datasets'}")
            write("\nChoose another run, or enter 0 at the mode menu to exit.")
    except (EOFError, KeyboardInterrupt):
        write("\nExiting.")
        return 0
