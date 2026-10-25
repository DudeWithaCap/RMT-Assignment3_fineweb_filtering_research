"""Join frozen qualitative judgments and validate the recorded language/dedup decisions.

No additional filtered dataset is created. The manual review is explicitly
AI-assisted and is not a gold-standard accuracy evaluation.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path

from src.filter_assets import ROOT, sha256_file
from src.filter_text import simplify_text
from src.filter_tokenizer import EnglishTokenizer
from src.language_filter import LanguageFilter
from src.run_filters import load_config
from .audit_stage3 import read_jsonl


def spotcheck():
    config = load_config(ROOT / "filter_config.json")
    output = ROOT / "results/validation"
    baseline = ROOT / config["baseline"]
    before = sha256_file(baseline)
    documents = {row["document_id"]: row for row in read_jsonl(baseline)}
    selection = json.loads((output / "language_sample_selection.json").read_text(encoding="utf-8"))["sample"]
    review = json.loads((ROOT / "validation/language_review.json").read_text(encoding="utf-8"))
    assert len(selection) == len(review["judgments"]) == 30
    assert before == review["baseline_sha256"], "Qualitative judgments belong to a different baseline"
    saved = {row["document_id"]: row for row in read_jsonl(ROOT / "results/language_decisions.jsonl")}
    model = LanguageFilter(config["language"], ROOT / config["assets_directory"])
    enriched, mismatches, decision_disagreement, top_disagreement = [], [], [], []
    counts, per_stratum = Counter(), defaultdict(Counter)
    for selection_row, judgment in zip(selection, review["judgments"]):
        identity = selection_row["document_id"]
        assert identity == judgment["document_id"], "Review selection changed; do not reuse old judgments"
        old = saved[identity]
        kept, reason, details = model.evaluate(documents[identity])
        if (kept, reason) != (old["kept"], old["reason"]) or details != old["details"]:
            mismatches.append(identity)
        label = judgment["label"]
        counts[label] += 1
        per_stratum[selection_row["stratum"]][label] += 1
        row = {**judgment, **selection_row, "predicted_language": old["details"]["language"],
               "confidence": old["details"]["confidence"], "filter_kept": old["kept"]}
        if label in ("english", "non_english"):
            target = label == "english"
            row["binary_top_language_agrees_with_review"] = (old["details"]["language"] == "en") == target
            row["gate_agrees_with_review_language_only"] = old["kept"] == target
            if not row["binary_top_language_agrees_with_review"]:
                top_disagreement.append(identity)
            if not row["gate_agrees_with_review_language_only"]:
                decision_disagreement.append(identity)
        enriched.append(row)
    result = {"reviewer": review["reviewer"], "review_basis": "Fixed stratified sample; excerpt-based AI-assisted judgments frozen before prediction join",
              "sample_count": 30, "judgment_counts": dict(counts), "strata": {key: dict(value) for key, value in per_stratum.items()},
              "determinate_binary_judgments": counts["english"] + counts["non_english"],
              "top_language_binary_disagreements": top_disagreement,
              "language_only_gate_disagreements": decision_disagreement,
              "saved_vs_reinferred_mismatches": mismatches,
              "limitation": "Not independent human ground truth, not a representative accuracy estimate, not calibration validation; confidence rejection of an English document is not a wrong top-language prediction",
              "judgments": enriched}
    (output / "language_spotcheck.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("LANGUAGE", json.dumps({key: value for key, value in result.items() if key != "judgments"}, indent=2))
    for row in enriched:
        print(row["review_index"], row["label"], row["stratum"], row["predicted_language"], round(row["confidence"], 4), row["filter_kept"])

    tokenizer = EnglishTokenizer(ROOT / config["assets_directory"])
    records = read_jsonl(ROOT / "results/dedup_removed.jsonl")
    dedup = {"removed_count": len(records), "match_types": dict(Counter(row["details"]["match_type"] for row in records)),
             "raw_exact_witness_pairs": 0, "normalized_exact_but_raw_different": 0, "witness_mismatches": [],
             "transitive_records": sum(row["details"].get("transitive_cluster", False) for row in records), "pairs": []}

    def grams(text):
        words = tokenizer.word_tokenize(simplify_text(text))
        n = config["dedup"]["ngram_size"]
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    for row in records:
        identity = row["document_id"]
        details = row["details"]
        first = documents[identity]["text"]
        second = documents[details["matched_document_id"]]["text"]
        raw_equal = first == second
        dedup["raw_exact_witness_pairs"] += raw_equal
        if details["match_type"] == "normalized_exact":
            valid = simplify_text(first) == simplify_text(second)
            dedup["normalized_exact_but_raw_different"] += not raw_equal
            measured = 1.0
        else:
            a, b = grams(first), grams(second)
            measured = len(a & b) / len(a | b) if a or b else 0
            valid = measured >= config["dedup"]["similarity_threshold"] and abs(measured - details["similarity"]) < 1e-12
        if not valid:
            dedup["witness_mismatches"].append(identity)
        dedup["pairs"].append({"document_id": identity, "matched_document_id": details["matched_document_id"],
                               "representative_id": details["representative_id"], "match_type": details["match_type"],
                               "measured_similarity": measured, "raw_exact": raw_equal})
    assert before == sha256_file(baseline)
    dedup["baseline_unchanged"] = True
    (output / "dedup_spotcheck.json").write_text(json.dumps(dedup, indent=2) + "\n", encoding="utf-8")
    print("DEDUP", json.dumps({key: value for key, value in dedup.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    spotcheck()
