# FineWeb Filtering Research

A CPU-only Python project with independent and sequential FineWeb-inspired filtering of real Common Crawl text: URL filtering, fastText English identification, heuristic quality checks and n-gram deduplication.

There are three prepared input datasets. Independent mode applies each filter separately to one selected input dataset and saves four filtered versions of it, one per filter. Sequential mode runs that input dataset through URL -> language -> quality -> deduplication and produces one final dataset passing all four filters. Document IDs, original text, URLs, domains and source metadata are preserved. Use independent mode for the original research comparisons and sequential mode for cumulative cleaning.

## Results: one folder per dataset and run

Every filtering run creates a new folder using the **start date, dataset name, filtering mode and start time** in Asia/Qyzylorda (UTC+05:00). Repeating a run creates another folder, even within the same second.

```text
results/
  2026-11-05_CC_2025_30_sequential_14-30-00/
    retention.png           Chart: retained and removed document counts
    retention.csv           The chart's numbers and percentages
    summary.json            Status, input hash, timestamps and removal reasons
    cleaned.json            Final dataset passing all four filters
    details/                Filter configuration, original JSONL outputs and logs
  archive/                  Previous results layout and historical analysis
```

Independent runs contain `datasets/url.json`, `language.json`, `quality.json` and `dedup.json` instead of one `cleaned.json`. Each JSON dataset is an array of complete document objects. Sequential charts show cumulative losses; the CSV also records losses at each individual step and percentages relative to the original baseline. Empty final outputs are `[]`. Later stages receiving zero documents are explicitly marked `skipped_empty_input`. Quality exports preserve original retained text; C4 working-text cleanup remains a separate measurement. Internal JSONL checkpoints and decision logs stay in `details/`; the optional `--keep-intermediate` flag additionally exports all four intermediate datasets as JSON.

All charts, measurements, cleaned datasets and decision logs stay local under `results/`. The entire `results/` and legacy `runs/` folders are Git-ignored, together with the frozen language-review results. The repository contains code and documentation rather than experimental outputs.

These exports are labeled `reorganized_existing_results`; they do not claim a new experiment. Historical acquisition records, full metric tables and research figures use `results/archive/results/` and `results/archive/runs/`. Those optional archive files are separate from the dated filtering runs and must be restored if removed.

## Setup and execution on Windows

Use Python 3.13, then run PowerShell from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.filter_assets
```

Start the interactive terminal menu:

```powershell
.\.venv\Scripts\python.exe main.py
```

Choose `1` for Independent or `2` for Sequential, then select dataset `1`, `2`, `3`, all three, or a custom JSONL file. Enter `0` to go back or exit. The menu waits for your input, runs all four filters, prints the output folder and returns to the mode menu. Invalid choices are requested again.

For scripting, run all four filters directly:

```powershell
.\.venv\Scripts\python.exe -m src.filter_datasets --dataset 1 --mode sequential
```

Use `--dataset 2`, `--dataset 3`, or `--dataset all`. Omit `--mode` or use `--mode independent` for independent filtering. Sequential mode always runs all four filters in the fixed order. Independent mode also supports selecting one filter with `--stage url`, `language`, `quality`, or `dedup`. The command prints the new results folder. Reuse the same command to create a separate dated run.

For a different prepared dataset:

```powershell
.\.venv\Scripts\python.exe -m src.filter_datasets --input data/processed/my_baseline.jsonl --dataset-name CC_2026_42 --mode sequential
```

Custom inputs must use the existing baseline JSONL schema: unique `document_id`, `text`, `url`, `domain`, character counts, truncation flag and Common Crawl `source` metadata. The document count is detected automatically; text must already respect the 20,000-character limit. Deduplication's configured size limit still applies. A crawl-based name is inferred when `--dataset-name` is omitted and the source metadata identifies exactly one crawl.

Unit tests use small temporary fixtures and do not create permanent experiment folders:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Cross-filter overlap analysis

Independent runs record every filter's keep/remove decision for every baseline document. The overlap analysis reads those decision logs (no filter is re-run) and measures how far the four filters remove the **same** documents:

```powershell
# Newest complete independent run for each dataset
.\.venv\Scripts\python.exe -m src.overlap --latest
# Specific runs; --no-profiles skips the text measurements (no tokenizer assets needed)
.\.venv\Scripts\python.exe -m src.overlap --runs results/<independent-run-1> results/<independent-run-2> --no-profiles
```

It is also menu option `3`. Each analysis creates `results/YYYY-MM-DD_cross_filter_overlap_HH-MM-SS/` with one subfolder per dataset:

| File | Contents |
| --- | --- |
| `pairs.csv` | For each filter pair: intersection/union counts, Jaccard, containment both ways, lift over independence, phi, and bootstrap 95% intervals |
| `per_filter.csv` | Removals per filter, removals no other filter makes, and the share also removed elsewhere |
| `patterns.csv`, `removal_patterns.png` | All 16 keep/remove combinations (UpSet-style), including documents every filter keeps |
| `overlap_heatmaps.png` | Jaccard, containment and phi matrices |
| `profiles.csv`, `removed_sentence_cdf.png` | Structural profile of each filter's removed, exclusively removed and retained sets: tokens, sentence-length statistics, unique domains, domain entropy, Wasserstein distance to baseline |
| `overlap.json` | Everything above plus source-run paths and verified hashes |

Combined `*_all_datasets.csv` tables sit at the top level. Before computing anything, the analysis verifies every decision log against its recorded SHA256. It also checks that all four filters judged the same baseline with identical document order, and that the local baseline still matches its recorded hash. Interpretation notes:

- Containment and Jaccard rise mechanically with a filter's removal rate (a filter that removes 95% "contains" most other removals by chance). Use lift and phi for chance-corrected association.
- Deduplication judges the whole input, so its overlap reflects which copy it kept.
- Intervals resample documents within one bounded sample; they are not population claims about Common Crawl.

## One-million-document experiment

The 1M sample repeats the Dataset 1 procedure at scale: CC-MAIN-2025-30, seed 42, WET files in SHA256(seed:path) order, consecutive eligible records. Its first 1,000 lines equal `baseline.jsonl`, so the 1K and 1M results come from nested samples.

```powershell
# Resumable download (about 43 WET files); rerun the same command after an interruption
.\.venv\Scripts\python.exe -m src.datasets.prepare_large --config config_1m.json --output-name baseline_1m.jsonl
# Independent filtering with the scalable dedup configuration
.\.venv\Scripts\python.exe -m src.filter_datasets --input data/processed/baseline_1m.jsonl --dataset-name CC_2025_30_1M --config filter_config_1m.json --mode independent
# Overlap analysis of that run
.\.venv\Scripts\python.exe -m src.overlap --runs results/<the new CC_2025_30_1M independent folder>
```

Downloaded WET files are cached as verified parts under `data/raw/baseline_1m/parts/`. Unlike the original script, a record that fails document validation is skipped and counted (with example URLs in the metadata) rather than stopping the download.

`filter_config_1m.json` matches `filter_config.json` except for deduplication. Exact pairwise comparison does not scale past a few thousand documents (it is now limited to 5,000), so the 1M run uses `method: minhash_lsh` ([src/dedup_lsh.py](src/dedup_lsh.py)):

- Same normalization, tokenizer and word 5-grams; normalized-exact matches are joined directly.
- MinHash (20 bands × 5 rows, seeded splitmix64 over 64-bit gram hashes) only proposes candidate pairs.
- Every candidate is checked with exact Jaccard ≥ 0.80, and connected components keep their earliest row, as before.
- A pair at Jaccard 0.80 is missed with probability about 0.03%. Very large buckets are checked as a star plus a chain.

`python -m validation.dedup_lsh_agreement` compares both methods on the three 1K baselines. It found identical decisions for all 3,000 documents.

## Code and inputs

- `src/prepare_dataset.py`: streaming acquisition for Dataset 1.
- `src/datasets/`: additional dataset acquisition, sampling, configuration helpers and validation (`prepare.py`, `common.py`, `run.py`, `summarize.py`).
- `main.py` and `src/interactive.py`: interactive terminal menu.
- `src/filter_datasets.py`: scriptable dated-run entry point; `src/run_filters.py`: filter execution and decision logs.
- `src/*_filter.py`: the four filter implementations; `src/run_results.py`: JSON exports and charts.
- `src/metrics/`: document/token counts, sentence statistics and domain diversity (`collect.py`, `calculations.py`, `verify.py`).
- `src/analysis/`: descriptive comparisons, hypothesis evidence and research figures (`analyze.py`, `statistics.py`, `plots.py`, `verify.py`).
- `config.json`, `stage35_config.json`, `filter_config.json`: acquisition settings and filtering thresholds.

The package moves replace the old `stage35/`, `stage4/` and `stage5/` source folders. Start the menu with `python main.py`, or use `python -m src.filter_datasets` for scripting. Additional module commands are `python -m src.datasets.prepare --help`, `python -m src.metrics.collect --help`, `python -m src.analysis.analyze --help` and `python -m src.analysis.plots --help`. Historical acquisition/metrics/analysis commands require their original manifests and inputs; if `results/archive/` is absent, restore those artifacts before running them. Optional evidence-dependent tests skip unavailable files. Old stage names inside saved provenance paths and `stage35_config.json` remain for compatibility; configurations and recorded results are not rewritten.

Existing baselines are `data/processed/baseline.jsonl`, `baseline_2.jsonl` and `baseline_3.jsonl`. Web-text datasets, models and all experimental results remain local and Git-ignored. Cloning provides the code and configuration; filtering additionally requires the baseline files and assets. Acquisition streams official Common Crawl WET data; synthetic research data is never substituted.

Sampling is bounded and nonrepresentative. WET extraction, compact fastText, exact five-gram Jaccard deduplication instead of MinHash/LSH, and English-oriented tokenization are documented adaptations of the [FineWeb methodology](https://arxiv.org/abs/2406.17557). Findings are descriptive, without population significance claims.
