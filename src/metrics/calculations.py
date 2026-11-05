"""Pure, deterministic descriptive metrics with bounded sentence histograms."""

from collections import Counter
import math


SENTENCE_FIELDS = ("sentence_mean_tokens", "sentence_median_tokens", "sentence_std_population_tokens",
                   "sentence_p25_tokens", "sentence_p75_tokens")


def safe_percentage(numerator, denominator):
    return 100.0 * numerator / denominator if denominator != 0 else None


def entropy_bits(counts):
    values = list(counts.values())
    if any(type(count) is not int or count < 0 for count in values):
        raise ValueError("Domain frequencies must be nonnegative integers")
    total = sum(values)
    if total == 0:
        return 0.0
    return math.fsum(-(count / total) * math.log2(count / total) for count in sorted(values) if count)


def histogram_quantile(histogram, probability):
    if not 0 <= probability <= 1:
        raise ValueError("Probability must lie in [0,1]")
    count = sum(histogram.values())
    if not count:
        return None
    position = (count - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    found, offset = {}, 0
    for length, frequency in sorted(histogram.items()):
        for rank in (lower, upper):
            if offset <= rank < offset + frequency:
                found[rank] = length
        offset += frequency
        if lower in found and upper in found:
            return found[lower] + (position - lower) * (found[upper] - found[lower])
    raise ValueError("Invalid sentence histogram")


def sentence_statistics(histogram):
    count = sum(histogram.values())
    if not count:
        return {"sentence_count": 0, **{field: None for field in SENTENCE_FIELDS}}
    mean = math.fsum(length * frequency for length, frequency in sorted(histogram.items())) / count
    variance = math.fsum(frequency * (length - mean) ** 2 for length, frequency in sorted(histogram.items())) / count
    return {"sentence_count": count, "sentence_mean_tokens": mean,
            "sentence_median_tokens": histogram_quantile(histogram, .5),
            "sentence_std_population_tokens": math.sqrt(variance),
            "sentence_p25_tokens": histogram_quantile(histogram, .25),
            "sentence_p75_tokens": histogram_quantile(histogram, .75)}


class MeasurementTokenizer:
    """Same English Punkt + NLTKWordTokenizer path used in existing quality rules."""

    def __init__(self, assets_directory):
        from src.filter_tokenizer import EnglishTokenizer
        self.tokenizer = EnglishTokenizer(assets_directory)

    def sentence_lengths(self, text):
        lengths = []
        for sentence in self.tokenizer.sent_tokenize(text):
            length = sum(bool(word.strip()) for word in self.tokenizer.words.tokenize(sentence))
            if length:
                lengths.append(length)
        return tuple(lengths)


class Accumulator:
    def __init__(self):
        self.document_count = 0
        self.token_count = 0
        self.character_count = 0
        self.domains = Counter()
        self.lengths = Counter()

    def add(self, domain, sentence_lengths, characters=0):
        if not isinstance(domain, str) or not domain:
            raise ValueError("Every document must have domain metadata")
        if any(type(length) is not int or length <= 0 for length in sentence_lengths):
            raise ValueError("Sentence token lengths must be positive integers")
        self.document_count += 1
        self.token_count += sum(sentence_lengths)
        self.character_count += characters
        self.domains[domain] += 1
        self.lengths.update(sentence_lengths)

    def summarize(self):
        return {"document_count": self.document_count, "token_count": self.token_count,
                "character_count": self.character_count, **sentence_statistics(self.lengths),
                "unique_domains": len(self.domains), "domain_entropy_bits": entropy_bits(self.domains)}


def relative_to_baseline(current, baseline):
    result = dict(current)
    result.update({"baseline_document_count": baseline["document_count"], "baseline_token_count": baseline["token_count"]})
    for unit in ("document", "token"):
        reduction = baseline[unit + "_count"] - current[unit + "_count"]
        result[unit + "_reduction_absolute"] = reduction
        result[unit + "_reduction_percentage"] = safe_percentage(reduction, baseline[unit + "_count"])
    result["domain_retention_percentage"] = safe_percentage(current["unique_domains"], baseline["unique_domains"])
    for key in ("sentence_count", *SENTENCE_FIELDS, "unique_domains", "domain_entropy_bits"):
        value, original = current[key], baseline[key]
        result[key + "_delta_from_baseline"] = value - original if value is not None and original is not None else None
    return result
