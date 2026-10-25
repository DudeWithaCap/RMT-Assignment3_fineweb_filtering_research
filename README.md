# FineWeb Filtering Research

A CPU-only Python project comparing four independent, FineWeb-inspired filters on three real Common Crawl samples (1,000 documents each).

Each filter reads the same original baseline: URL rules, fastText English identification, heuristic quality checks, or n-gram deduplication. Filters are **not sequential**. Measurements cover document/token reduction, sentence lengths, and domain diversity. Baselines are never overwritten.

## Structure and saved results

- `src/`: dataset preparation, filter implementations and runners.
- `stage35/`: additional sampling and cross-dataset validation.
- `stage4/`: metric collection; `stage5/`: descriptive analysis and figures.
- `tests/`: automated correctness and reproducibility checks.
- `data/`: local datasets, source caches and downloaded assets (Git-ignored).
- `results/`: Dataset 1 filter logs, provenance, combined tables and figures.
- `runs/stage35/primary/dataset_2/results/` and `dataset_3/results/`: Dataset 2 and 3 filter logs and summaries.

Start with [filtering results](results/stage35/filtering_results.csv), [combined metrics](results/stage4/combined_summary.csv), [hypothesis assessments](results/stage5/hypothesis_assessments.csv), and [figures](results/stage5/figures/).

| Dataset | Crawl | Baseline | URL kept | English kept | Quality kept | Dedup kept |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | CC-MAIN-2025-30 | 1,000 | 992 | 207 | 47 | 971 |
| 2 | CC-MAIN-2025-18 | 1,000 | 993 | 216 | 53 | 992 |
| 3 | CC-MAIN-2025-47 | 1,000 | 995 | 230 | 41 | 958 |

## Windows setup

Use Python 3.13 (the tested version), then run PowerShell from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Using the environment's executable directly avoids activation-policy problems. Tests do not download assets; tests requiring absent local datasets/models are skipped.

## Run filters

Full filtering requires the original local files `data/processed/baseline.jsonl`, `baseline_2.jsonl`, and `baseline_3.jsonl`. Web-text datasets and models are deliberately excluded from Git; cloning the repository alone provides the saved results, not these inputs.

```powershell
.\.venv\Scripts\python.exe -m src.filter_assets
.\.venv\Scripts\python.exe -m src.filter_datasets --dataset all --stage all
```

New outputs go to `runs/reproduction/dataset_1/`, `dataset_2/`, and `dataset_3/`. Existing output directories are rejected. To run one condition or repeat safely:

```powershell
.\.venv\Scripts\python.exe -m src.filter_datasets --dataset 2 --stage quality --output-dir runs/reproduction/quality_check
```

Settings are in `config.json`, `stage35_config.json`, and `filter_config.json`; asset checksums are pinned in `filter_assets.lock.json`. Acquisition implementations are `src.prepare_dataset` and `stage35.prepare`. Reacquisition needs network access and can rewrite provenance records; use a separate working copy. No synthetic dataset is substituted.

## Reproduce analysis and plots

These commands use the committed metrics and write separate outputs:

```powershell
.\.venv\Scripts\python.exe -m stage5.analyze --output-dir results/stage5/review_replay
.\.venv\Scripts\python.exe -m stage5.plots --analysis-dir results/stage5/review_replay --output-dir results/stage5/review_replay/figures
```

To remeasure the original experimental conditions, all original baseline and filtered JSONL files plus cached upstream validation sources must be available locally:

```powershell
.\.venv\Scripts\python.exe -m stage4.collect --output-dir results/stage4/review_replay
```

Stage 4 reads the original condition locations, not `runs/reproduction/`.

## Methodological limits

Dataset 1 is an archive-prefix sample; Datasets 2 and 3 use seeded hash selection within bounded WET windows. These are not representative random samples of Common Crawl. Text is capped at 20,000 characters. WET extraction, compact fastText, exact five-gram Jaccard deduplication (rather than MinHash/LSH), and English-oriented tokenization differ from the [original FineWeb pipeline](https://arxiv.org/abs/2406.17557).

Primary quality metrics use saved original text; C4 text-cleaning effects are measured separately. Findings are descriptive, without population significance claims. Historical manifests remain unchanged; current preservation checks protect research artifacts while allowing source/documentation maintenance. ZIP archives, replay copies and assignment packaging remain local and are Git-ignored.
