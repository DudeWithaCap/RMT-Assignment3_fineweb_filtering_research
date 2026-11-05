"""Compare exact and MinHash/LSH deduplication decisions on the prepared baselines.

Both filters read the same baseline rows with the same tokenizer and threshold. The
report lists every document whose keep/remove decision or reason differs. Run:

    python -m validation.dedup_lsh_agreement
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import tempfile
import time

from src.datasets.common import baseline_path
from src.dedup_filter import DedupFilter
from src.dedup_lsh import LSHDedupFilter
from src.filter_assets import ROOT, sha256_file
from src.filter_tokenizer import EnglishTokenizer
from src.run_filters import baseline_rows, load_config


def compare(baseline, exact_config, lsh_config, assets):
    tokenizer = EnglishTokenizer(assets)
    rows = lambda: baseline_rows(baseline, 20000)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        exact = DedupFilter(exact_config, tokenizer, Path(directory))
        for _, document in rows():
            exact.add(document)
        exact_decisions = exact.decisions()
        exact.close()
    exact_seconds = time.perf_counter() - started
    started = time.perf_counter()
    lsh = LSHDedupFilter(lsh_config, partial(EnglishTokenizer, assets), tokenizer.asset_record)
    lsh_decisions = lsh.decide_stream(rows)
    lsh_seconds = time.perf_counter() - started
    differences = [{"document_id": identity, "exact": exact_decisions[identity][:2], "lsh": lsh_decisions[identity][:2]}
                   for identity in exact_decisions if exact_decisions[identity][:2] != lsh_decisions[identity][:2]]
    return {"baseline": str(baseline), "baseline_sha256": sha256_file(baseline), "documents": len(exact_decisions),
            "exact_removed": sum(not d[0] for d in exact_decisions.values()),
            "lsh_removed": sum(not d[0] for d in lsh_decisions.values()),
            "differences": differences, "identical": not differences, "lsh_statistics": lsh.statistics,
            "exact_seconds": exact_seconds, "lsh_seconds": lsh_seconds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baselines", nargs="+", type=Path,
                        default=[baseline_path(d) for d in (1, 2, 3)])
    parser.add_argument("--output", type=Path, default=ROOT / "results/dedup_lsh_agreement.json")
    args = parser.parse_args(argv)
    exact_config = load_config(ROOT / "filter_config.json")
    lsh_config = load_config(ROOT / "filter_config_1m.json")
    assets = ROOT / exact_config["assets_directory"]
    comparisons = []
    for baseline in args.baselines:
        result = compare(baseline, exact_config["dedup"], lsh_config["dedup"], assets)
        print(f"{baseline.name}: exact removed {result['exact_removed']}, LSH removed {result['lsh_removed']}, "
              f"{len(result['differences'])} differing decisions", flush=True)
        comparisons.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                       "exact_config": exact_config["dedup"], "lsh_config": lsh_config["dedup"],
                                       "comparisons": comparisons}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    return 0 if all(result["identical"] for result in comparisons) else 1


if __name__ == "__main__":
    raise SystemExit(main())
