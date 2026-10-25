"""Analyze the fixed Stage 4 pilot; no population p-values or new filtering."""

import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
import math
from pathlib import Path

from src.filter_assets import ROOT, sha256_file
from stage4.collect import write_csv, write_json
from .preserve import OUTPUT, verify_preserved
from .statistics import (descriptive, distribution_distances, expected_unique_domains, expected_entropy_same_size,
                         ratio, retention_discordance)

STAGE4 = ROOT / "results/stage4"
FILTERS = ("url", "language", "quality", "dedup")
CONDITIONS = ("baseline", *FILTERS)


def read_inputs():
    manifest = json.loads((STAGE4 / "manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["output_sha256"].items():
        if sha256_file(STAGE4 / name) != checksum:
            raise ValueError(f"Stage 4 output changed: {name}")
    metrics = json.loads((STAGE4 / "metrics.json").read_text(encoding="utf-8"))
    supplements = json.loads((STAGE4 / "quality_c4_metrics.json").read_text(encoding="utf-8"))
    if len(metrics) != 15 or len(supplements) != 3:
        raise ValueError("Expected all 15 primary and three supplemental conditions")
    indexed = {(row["dataset"], row["condition"]): row for row in metrics + supplements}
    histograms, domains, documents = defaultdict(Counter), defaultdict(Counter), defaultdict(dict)
    with (STAGE4 / "sentence_histograms.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            histograms[(int(row["dataset"]), row["condition"])][int(row["sentence_tokens"])] = int(row["frequency"])
    with (STAGE4 / "domain_counts.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            domains[(int(row["dataset"]), row["condition"])][row["domain"]] = int(row["documents"])
    with (STAGE4 / "document_metrics.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            row = dict(row)
            for key in ("dataset", "token_count", "sentence_count", "character_count", "source_record_index"):
                row[key] = int(row[key])
            key = row["dataset"], row["condition"]
            if row["document_id"] in documents[key]:
                raise ValueError("Repeated document ID in Stage 4 intermediates")
            documents[key][row["document_id"]] = row
    for key, metric in indexed.items():
        if (sum(histograms[key].values()) != metric["sentence_count"]
                or sum(length * n for length, n in histograms[key].items()) != metric["token_count"]
                or len(documents[key]) != metric["document_count"]
                or sum(domains[key].values()) != metric["document_count"]):
            raise ValueError(f"Inconsistent Stage 4 intermediates: {key}")
    return indexed, histograms, domains, documents


def document_mean_lengths(documents):
    return [row["token_count"] / row["sentence_count"] for row in documents.values() if row["sentence_count"]]


def analyze(directory=OUTPUT, overwrite=False):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(OUTPUT.resolve()):
        raise ValueError("Analysis outputs must stay under results/stage5")
    if (directory / "analysis.json").exists() and not overwrite:
        raise FileExistsError("Analysis already exists; choose a new --output-dir for replay")
    protected = verify_preserved()
    directory.mkdir(parents=True, exist_ok=True)
    metrics, histograms, domains, documents = read_inputs()
    effects, size_adjustment, pairwise, h1, c4_pairs, c4_summaries, unit_counts = [], [], [], [], [], [], []
    for dataset in (1, 2, 3):
        base_key = dataset, "baseline"
        baseline, base_docs = metrics[base_key], documents[base_key]
        baseline_document_means = descriptive(document_mean_lengths(base_docs))
        unit_counts.append({"dataset": dataset, "baseline_documents": len(base_docs),
                            "baseline_hostnames": len(domains[base_key]),
                            "source_wet_windows": len({row["source_wet_path"] for row in base_docs.values()}),
                            "baseline_sentence_bearing_documents": baseline_document_means["n"]})
        removed_sets = {}
        for condition in FILTERS:
            key = dataset, condition
            row, current_docs = metrics[key], documents[key]
            if not set(current_docs) <= set(base_docs):
                raise ValueError("Condition uses documents outside its baseline")
            for identity, document in current_docs.items():
                if any(document[field] != base_docs[identity][field] for field in
                       ("domain", "token_count", "sentence_count", "character_count", "source_wet_path", "source_record_index")):
                    raise ValueError("Primary condition modifies a retained document's measurements")
            removed_sets[condition] = set(base_docs) - set(current_docs)
            retained_document_means = descriptive(document_mean_lengths(current_docs))
            removed_document_means = descriptive(document_mean_lengths({identity: base_docs[identity] for identity in removed_sets[condition]}))
            baseline_iqr = baseline["sentence_p75_tokens"] - baseline["sentence_p25_tokens"]
            current_iqr = row["sentence_p75_tokens"] - row["sentence_p25_tokens"]
            baseline_cv = ratio(baseline["sentence_std_population_tokens"], baseline["sentence_mean_tokens"])
            current_cv = ratio(row["sentence_std_population_tokens"], row["sentence_mean_tokens"])
            effect = {"dataset": dataset, "filter": condition,
                      "baseline_documents": baseline["document_count"], "retained_documents": row["document_count"],
                      "document_reduction_percentage": row["document_reduction_percentage"],
                      "token_reduction_percentage": row["token_reduction_percentage"],
                      "sentence_mean_change_tokens": row["sentence_mean_tokens"] - baseline["sentence_mean_tokens"],
                      "sentence_mean_ratio": ratio(row["sentence_mean_tokens"], baseline["sentence_mean_tokens"]),
                      "sentence_median_change_tokens": row["sentence_median_tokens"] - baseline["sentence_median_tokens"],
                      "sentence_sd_ratio": ratio(row["sentence_std_population_tokens"], baseline["sentence_std_population_tokens"]),
                      "baseline_iqr_tokens": baseline_iqr, "retained_iqr_tokens": current_iqr, "sentence_iqr_ratio": ratio(current_iqr, baseline_iqr),
                      "baseline_coefficient_of_variation": baseline_cv, "retained_coefficient_of_variation": current_cv,
                      "sentence_cv_ratio": ratio(current_cv, baseline_cv),
                      **distribution_distances(histograms[base_key], histograms[key]),
                      "unique_domain_loss_percentage": 100 - row["domain_retention_percentage"],
                      "entropy_loss_bits": baseline["domain_entropy_bits"] - row["domain_entropy_bits"],
                      "domain_retention_percentage": row["domain_retention_percentage"],
                      "baseline_document_weighted_mean_sentence_tokens": baseline_document_means["mean"],
                      "retained_document_weighted_mean_sentence_tokens": retained_document_means["mean"],
                      "removed_document_weighted_mean_sentence_tokens": removed_document_means["mean"],
                      "retained_sentence_bearing_documents": retained_document_means["n"],
                      "removed_sentence_bearing_documents": removed_document_means["n"],
                      "p_value": None, "inference_status": "descriptive_fixed_sample_only"}
            effects.append(effect)
            expected = expected_unique_domains(domains[base_key], row["document_count"])
            expected_entropy = expected_entropy_same_size(domains[base_key], row["document_count"])
            size_adjustment.append({"dataset": dataset, "filter": condition, "retained_documents": row["document_count"],
                                    "observed_unique_domains": row["unique_domains"], "expected_unique_uniform_same_k": expected,
                                    "observed_minus_expected_domains": row["unique_domains"] - expected,
                                    "observed_over_expected_domains": ratio(row["unique_domains"], expected),
                                    "observed_entropy_bits": row["domain_entropy_bits"],
                                    "expected_entropy_uniform_same_k": expected_entropy,
                                    "observed_minus_expected_entropy_bits": row["domain_entropy_bits"] - expected_entropy,
                                    "maximum_entropy_at_observed_unique_domains": math.log2(row["unique_domains"]) if row["unique_domains"] else 0,
                                    "benchmark_type": "exact_finite_baseline_uniform_k_subset_expectation_not_a_test", "p_value": None})
        for first, second in combinations(FILTERS, 2):
            paired = retention_discordance(removed_sets[first], removed_sets[second], base_docs)
            difference_tokens = sum(base_docs[identity]["token_count"] for identity in removed_sets[first]) - sum(base_docs[identity]["token_count"] for identity in removed_sets[second])
            pairwise.append({"dataset": dataset, "first_filter": first, "second_filter": second, **paired,
                             "token_reduction_difference_pp": 100 * difference_tokens / baseline["token_count"],
                             "p_value": None, "comparison": "same_document_ids_paired_descriptive_contrast"})
        for target in ("quality", "dedup"):
            for comparator in ("url", "language"):
                a, b = metrics[(dataset, target)], metrics[(dataset, comparator)]
                h1.append({"dataset": dataset, "target": target, "comparator": comparator,
                           "document_reduction_difference_pp": a["document_reduction_percentage"] - b["document_reduction_percentage"],
                           "token_reduction_difference_pp": a["token_reduction_percentage"] - b["token_reduction_percentage"],
                           "target_greater_document_reduction": a["document_reduction_percentage"] > b["document_reduction_percentage"],
                           "target_greater_token_reduction": a["token_reduction_percentage"] > b["token_reduction_percentage"], "p_value": None})
        saved, cleaned = documents[(dataset, "quality")], documents[(dataset, "quality_c4_transformed")]
        if set(saved) != set(cleaned):
            raise ValueError("C4 before/after IDs are not paired")
        token_changes, mean_changes = [], []
        for identity in sorted(saved):
            a, b = saved[identity], cleaned[identity]
            if a["domain"] != b["domain"]:
                raise ValueError("C4 changed domain metadata")
            old_mean, new_mean = ratio(a["token_count"], a["sentence_count"]), ratio(b["token_count"], b["sentence_count"])
            change = b["token_count"] - a["token_count"]
            mean_change = new_mean - old_mean if old_mean is not None and new_mean is not None else None
            token_changes.append(change)
            if mean_change is not None:
                mean_changes.append(mean_change)
            c4_pairs.append({"dataset": dataset, "document_id": identity, "domain": a["domain"],
                             "saved_tokens": a["token_count"], "cleaned_tokens": b["token_count"], "token_change": change,
                             "saved_sentences": a["sentence_count"], "cleaned_sentences": b["sentence_count"],
                             "sentence_count_change": b["sentence_count"] - a["sentence_count"],
                             "saved_mean_sentence_tokens": old_mean, "cleaned_mean_sentence_tokens": new_mean,
                             "paired_mean_sentence_change_tokens": mean_change})
        totals = metrics[(dataset, "quality")], metrics[(dataset, "quality_c4_transformed")]
        c4_summaries.append({"dataset": dataset, "paired_documents": len(saved),
                             "saved_tokens": totals[0]["token_count"], "cleaned_tokens": totals[1]["token_count"],
                             "token_change": sum(token_changes), "token_reduction_percentage_of_saved_quality": 100 * -sum(token_changes) / totals[0]["token_count"],
                             "documents_tokens_decrease": sum(change < 0 for change in token_changes),
                             "documents_tokens_increase": sum(change > 0 for change in token_changes),
                             "documents_tokens_tie": sum(change == 0 for change in token_changes),
                             "paired_token_change_mean": descriptive(token_changes)["mean"],
                             "paired_token_change_median": descriptive(token_changes)["median"],
                             "paired_document_mean_sentence_change_mean": descriptive(mean_changes)["mean"],
                             "pooled_sentence_mean_change_tokens": totals[1]["sentence_mean_tokens"] - totals[0]["sentence_mean_tokens"],
                             **distribution_distances(histograms[(dataset, "quality")], histograms[(dataset, "quality_c4_transformed")]),
                             "domain_counts_unchanged": domains[(dataset, "quality")] == domains[(dataset, "quality_c4_transformed")], "p_value": None})
    cross_dataset = []
    for condition in FILTERS:
        rows = [row for row in effects if row["filter"] == condition]
        for field in ("document_reduction_percentage", "token_reduction_percentage", "cdf_sup_distance", "wasserstein1_tokens", "unique_domain_loss_percentage", "entropy_loss_bits"):
            cross_dataset.append({"filter": condition, "metric": field, **descriptive([row[field] for row in rows]),
                                  "scope": "three_heterogeneous_fixed_samples_equal_weight_not_population_estimate"})
    decisions = [
        {"candidate": "independent_sentence_t_mann_whitney_or_ks_tests", "performed": False, "p_value": None,
         "reason": "Discrete sentences nested within documents; filtered sets overlap baselines; independent-sample assumptions fail."},
        {"candidate": "paired_dataset_t_wilcoxon_or_friedman_tests", "performed": False, "p_value": None,
         "reason": "Only three nonrepresentative datasets, heterogeneous sampling, five archive windows and repeated content; not independent exchangeable population replicates."},
        {"candidate": "sentence_iid_or_population_bootstrap_confidence_intervals", "performed": False, "p_value": None,
         "reason": "Sentence IID would create pseudoreplication; only one/two archive windows per dataset cannot support defensible archive-level population uncertainty."},
        {"candidate": "paired_C4_sign_or_wilcoxon_test", "performed": False, "p_value": None,
         "reason": "Deterministic cleanup on selected nonrandom documents; no justified independent symmetric/random-sign null. Paired descriptive changes are reported."},
    ]
    hypotheses = [
        {"hypothesis": "H1", "status": "partially_supported", "scope": "descriptive_within_studied_samples",
         "supporting_evidence": "Quality reduces both documents and tokens more than URL and language in all three samples; dedup removes more documents than URL in all three.",
         "contradicting_evidence": "Dedup reduces less volume than language in every sample, and less token volume than URL in Datasets 2 and 3. The full conjunction is not supported.",
         "statistical_conclusion": "The original significantly-higher claim is not established; no valid population test was performed."},
        {"hypothesis": "H2", "status": "partially_supported", "scope": "descriptive_within_studied_samples",
         "supporting_evidence": "Quality medians rise and SD/IQR/CV decline in all three samples; dedup medians and quartiles are unchanged, with small observed CDF changes.",
         "contradicting_evidence": "Quality mean sentence lengths decrease in every sample. The longer-sentence claim depends on the location summary, not a uniform rightward shift.",
         "statistical_conclusion": "No distributional equivalence margin or valid inferential design; small dedup distances do not prove equality."},
        {"hypothesis": "H3", "status": "partially_supported", "scope": "raw_diversity_loss_supported_disproportionate_claim_qualified",
         "supporting_evidence": "Language and quality have much greater raw hostname/entropy loss than dedup across all three samples.",
         "contradicting_evidence": "Raw richness loss is confounded by retained size; quality survivors occupy distinct hostnames and do not lose richness beyond the same-size uniform-deletion expectation.",
         "statistical_conclusion": "No low-resource/non-standard-domain labels were defined or validated, so that original mechanism is inconclusive; no population significance is claimed."},
    ]
    result = {"sentence_effects": effects, "domain_size_adjustment": size_adjustment, "filter_contrasts": pairwise,
              "h1_contrasts": h1, "c4_paired_documents": c4_pairs, "c4_summary": c4_summaries,
              "cross_dataset_descriptive_summary": cross_dataset, "sampling_units": unit_counts,
              "statistical_test_decisions": decisions, "hypothesis_assessments": hypotheses,
              "inferential_p_values_produced": False, "prior_files_unchanged": protected}
    exported = [directory / "analysis.json"]
    for name, records in result.items():
        if isinstance(records, list):
            write_csv(directory / f"{name}.csv", records)
            write_json(directory / f"{name}.json", records)
            exported.extend((directory / f"{name}.csv", directory / f"{name}.json"))
    write_json(directory / "analysis.json", result)
    source_files = ("metrics.json", "quality_c4_metrics.json", "document_metrics.csv", "domain_counts.csv", "sentence_histograms.csv")
    write_json(directory / "analysis_manifest.json", {
        "inputs_sha256": {name: sha256_file(STAGE4 / name) for name in source_files},
        "output_sha256": {path.name: sha256_file(path) for path in sorted(exported)},
        "prior_files_unchanged": verify_preserved(), "statistical_scope": "fixed_sample_descriptive_no_population_p_values",
        "implementation_sha256": {path.name: sha256_file(path) for path in sorted((ROOT / "stage5").glob("*.py"))}})
    print(f"Analyzed 12 baseline contrasts, 18 paired filter contrasts and {len(c4_pairs)} paired C4 documents", flush=True)
    for row in effects:
        if row["filter"] in ("quality", "dedup"):
            print(row["dataset"], row["filter"], "CDF_D", round(row["cdf_sup_distance"], 6), "W1", round(row["wasserstein1_tokens"], 6), "SD_ratio", round(row["sentence_sd_ratio"], 4), flush=True)
    for row in size_adjustment:
        print(row["dataset"], row["filter"], "domains_observed_expected", row["observed_unique_domains"], round(row["expected_unique_uniform_same_k"], 4), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace Stage 5 analysis tables only; prior stages are protected")
    args = parser.parse_args()
    analyze(args.output_dir, args.overwrite)
