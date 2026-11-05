"""Scalable deduplication: MinHash/LSH candidate search, exact Jaccard verification.

The exact filter (dedup_filter.DedupFilter) compares every document with every earlier
document that shares a 5-gram, which does not scale past small samples. This variant
keeps the same decision rule and only changes how candidate pairs are found:

* identical normalization, tokenization and word 5-gram sets as the exact filter;
* normalized-text SHA256 matches are joined directly ("normalized_exact");
* MinHash signatures (bands x rows permutations over 64-bit gram hashes) with LSH
  banding propose candidate pairs; each candidate's Jaccard similarity is then
  computed exactly on the 64-bit gram hashes and joined only if >= the threshold;
* connected components keep their earliest input row, with the same decision
  explanations as the exact filter.

Differences from the exact filter, to be reported with results: a similar pair can be
missed if it never shares an LSH band (probability 1-(1-J^r)^b per pair, about 0.03%
at J = 0.8 for 20 bands x 5 rows); grams are compared by 64-bit hash rather than by
string; buckets larger than max_bucket_pairs are verified as a star plus a chain instead
of all pairs. Results do not depend on the number of worker processes.
"""

import hashlib
from multiprocessing import get_context
import os

import numpy as np

from .dedup_filter import DuplicateClusters
from .filter_text import simplify_text

_STATE = {}


def _init(tokenizer_factory, ngram_size, num_perm, seed):
    rng = np.random.default_rng(seed)
    _STATE.update(tokenizer=tokenizer_factory(), n=ngram_size,
                  seeds=rng.integers(0, np.iinfo(np.uint64).max, num_perm, dtype=np.uint64, endpoint=True))


def mix(hashes, seeds):
    """Seeded splitmix64 finalizer: one independent-looking 64-bit hash per permutation.

    A linear (a*x + b) mod p family over 32-bit inputs was tried first and rejected: its
    minima were dominated by the smallest inputs, so permutations were strongly correlated
    and similar documents agreed far less often than their Jaccard similarity.
    """
    with np.errstate(over="ignore"):
        z = hashes[:, None] ^ seeds
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _gram_hashes(normalized):
    words = _STATE["tokenizer"].word_tokenize(normalized)
    n = _STATE["n"]
    grams = {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    hashes = np.fromiter((int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
                          for g in grams), dtype=np.uint64, count=len(grams))
    return np.unique(hashes)


def _signature(item):
    index, text = item
    normalized = simplify_text(text)
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).digest()
    hashes = _gram_hashes(normalized)
    if not len(hashes):
        return index, fingerprint, None
    # Keep the low 32 bits of each minimum: equal minima stay equal, and unequal ones
    # collide with probability 2^-32, which only adds a candidate that is then verified.
    minima = mix(hashes, _STATE["seeds"]).min(axis=0)
    return index, fingerprint, (minima & np.uint64(0xFFFFFFFF)).astype(np.uint32).tobytes()


def _hashes(item):
    index, text = item
    return index, _gram_hashes(simplify_text(text))


def bucket_pairs(members, max_bucket_pairs):
    """Candidate pairs (low, high) for one LSH bucket of ascending row indices."""
    if len(members) <= max_bucket_pairs:
        first, second = np.triu_indices(len(members), 1)
        return members[first], members[second]
    # Very large buckets are near-identical boilerplate: compare to the earliest member and to
    # the previous member, which keeps components connected without quadratic cost.
    return (np.r_[np.repeat(members[0], len(members) - 1), members[1:-1]],
            np.r_[members[1:], members[2:]])


class LSHDedupFilter(DuplicateClusters):
    def __init__(self, config, tokenizer_factory, asset_record, workers=None):
        self.config, self.tokenizer_factory = config, tokenizer_factory
        self.bands, self.rows = config["num_bands"], config["rows_per_band"]
        self.num_perm = self.bands * self.rows
        configured = workers if workers is not None else config.get("workers")
        self.workers = configured or max(1, min(16, (os.cpu_count() or 2) - 2))
        self.parents, self.ids, self.edges = [], [], []
        self.assets = {"punkt_tab.zip": asset_record}
        self.statistics = {}

    def _map(self, function, items):
        arguments = (self.tokenizer_factory, self.config["ngram_size"], self.num_perm, self.config["seed"])
        if self.workers == 1:
            _init(*arguments)
            yield from map(function, items)
            return
        with get_context("spawn").Pool(self.workers, initializer=_init, initargs=arguments) as pool:
            yield from pool.imap(function, items, chunksize=128)

    def decide_stream(self, rows):
        """Two passes over rows(): signatures, then exact verification of LSH candidates."""
        signatures, signed, exact = bytearray(), [], {}
        documents = ((index, document["text"]) for index, (_, document) in enumerate(rows()))
        ids = []
        for index, (_, document) in enumerate(rows()):
            if index >= self.config["max_documents"]:
                raise ValueError("Dedup max_documents exceeded")
            ids.append(document["document_id"])
        self.ids, self.parents = ids, list(range(len(ids)))
        for index, fingerprint, signature in self._map(_signature, documents):
            if fingerprint in exact:
                # Same normalized text implies the same gram set: it needs no LSH entry.
                self.union(index, exact[fingerprint], 1.0, "normalized_exact")
                continue
            exact[fingerprint] = index
            if signature is not None:
                signatures += signature
                signed.append(index)
        total = len(ids)
        matrix = np.frombuffer(bytes(signatures), dtype=np.uint32).reshape(len(signed), self.num_perm)
        signed = np.array(signed, dtype=np.int64)
        multipliers = np.random.default_rng(self.config["seed"] + 1).integers(
            1, 1 << 63, self.rows, dtype=np.uint64) | np.uint64(1)
        codes = []
        for band in range(self.bands):
            block = matrix[:, band * self.rows:(band + 1) * self.rows].astype(np.uint64)
            keys = (block * multipliers).sum(axis=1)  # wraps modulo 2^64 by design
            order = np.argsort(keys, kind="stable")
            sorted_keys = keys[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
            ends = np.r_[starts[1:], len(sorted_keys)]
            for start, end in zip(starts[ends - starts > 1], ends[ends - starts > 1]):
                low, high = bucket_pairs(signed[order[start:end]], self.config["max_bucket_pairs"])
                codes.append(low * total + high)
        pairs = np.unique(np.concatenate(codes)) if codes else np.empty(0, dtype=np.int64)
        needed = np.zeros(total, dtype=bool)
        needed[pairs // total] = needed[pairs % total] = True
        grams = {}
        if len(pairs):
            wanted = ((index, document["text"]) for index, (_, document) in enumerate(rows()) if needed[index])
            grams = dict(self._map(_hashes, wanted))
        verified = 0
        for code in pairs:
            low, high = divmod(int(code), total)
            if self.find(low) == self.find(high):
                continue
            first, second = grams[low], grams[high]
            intersection = len(np.intersect1d(first, second, assume_unique=True))
            score = intersection / (len(first) + len(second) - intersection)
            verified += 1
            if score >= self.config["similarity_threshold"]:
                self.union(high, low, score, "ngram_jaccard")
        self.statistics = {"documents": total, "documents_with_signature": len(signed),
                           "candidate_pairs": int(len(pairs)), "documents_reverified": int(needed.sum()),
                           "pairs_compared": verified, "bands": self.bands, "rows_per_band": self.rows,
                           "workers": self.workers}
        self.assets["minhash_lsh"] = self.statistics
        return self.decisions()
