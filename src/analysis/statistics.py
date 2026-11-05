"""Descriptive effect sizes; intentionally no independent-sample p-values."""

from collections import Counter
import math


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def distribution_distances(first, second):
    """Exact empirical CDF sup distance and Wasserstein-1, from count histograms.

    They remain descriptive when samples overlap. No KS reference distribution,
    independence assumption, or p-value is attached to these distances.
    """
    a, b = sum(first.values()), sum(second.values())
    if not a or not b:
        return {"cdf_sup_distance": None, "wasserstein1_tokens": None}
    points = sorted(set(first) | set(second))
    ca = cb = maximum = area = 0.0
    for index, point in enumerate(points):
        ca += first.get(point, 0) / a
        cb += second.get(point, 0) / b
        difference = abs(ca - cb)
        maximum = max(maximum, difference)
        if index + 1 < len(points):
            area += difference * (points[index + 1] - point)
    return {"cdf_sup_distance": min(1.0, maximum), "wasserstein1_tokens": area}


def missing_probability(population, domain_count, retained):
    """P(no document from a domain) under uniform fixed-size subset selection."""
    if any(type(value) is not int for value in (population, domain_count, retained)):
        raise ValueError("Counts must be integers")
    if not 0 <= domain_count <= population or not 0 <= retained <= population:
        raise ValueError("Invalid finite-population counts")
    if domain_count == 0 or retained == 0:
        return 1.0
    if retained > population - domain_count:
        return 0.0
    # Equivalent to C(N-n_i,k)/C(N,k), but at most n_i stable factors.
    return math.prod((population - retained - j) / (population - j) for j in range(domain_count))


def expected_unique_domains(domain_counts, retained):
    if any(type(count) is not int or count < 0 for count in domain_counts.values()):
        raise ValueError("Domain counts must be nonnegative integers")
    population = sum(domain_counts.values())
    if not 0 <= retained <= population:
        raise ValueError("Retained count exceeds population")
    return math.fsum(1.0 - missing_probability(population, count, retained)
                     for count in sorted(domain_counts.values()) if count)


def expected_entropy_same_size(domain_counts, retained):
    """Exact E[empirical Shannon entropy] for a uniform k-subset of fixed records.

    For each hostname X follows Hypergeom(N,n_i,k). Entropy is log2(k) minus
    sum_i X_i*log2(X_i)/k. This is a mathematical reference, not a p-value.
    """
    expected_unique_domains(domain_counts, retained)  # validate counts/range
    population = sum(domain_counts.values())
    if retained == 0:
        return 0.0
    denominator = math.comb(population, retained)
    contributions = []
    for count, multiplicity in sorted(Counter(domain_counts.values()).items()):
        expected_xlogx = math.fsum(
            (math.comb(count, x) * math.comb(population - count, retained - x) / denominator) * x * math.log2(x)
            for x in range(max(2, retained - (population - count)), min(count, retained) + 1))
        contributions.append(multiplicity * expected_xlogx)
    return max(0.0, math.log2(retained) - math.fsum(contributions) / retained)


def retention_discordance(first_removed, second_removed, baseline_ids):
    first, second, baseline = set(first_removed), set(second_removed), set(baseline_ids)
    if not (first | second) <= baseline:
        raise ValueError("Unknown document in removal indicators")
    both = len(first & second)
    return {"removed_by_both": both, "removed_by_first_only": len(first - second),
            "removed_by_second_only": len(second - first), "kept_by_both": len(baseline - first - second),
            "document_reduction_difference_pp": 100 * (len(first) - len(second)) / len(baseline) if baseline else None}


def descriptive(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    middle = n // 2
    median = values[middle] if n % 2 else (values[middle - 1] + values[middle]) / 2
    return {"n": n, "mean": math.fsum(values) / n, "median": median, "minimum": values[0], "maximum": values[-1]}
