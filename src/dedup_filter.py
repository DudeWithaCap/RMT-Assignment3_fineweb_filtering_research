"""Deterministic exact Jaccard comparisons using disk-backed word n-gram sets.

This intentionally replaces FineWeb's approximate MinHash/LSH candidate search.
All threshold-passing edges form components; the earliest baseline row survives.
"""

import hashlib
import sqlite3

from .filter_text import simplify_text


class DedupFilter:
    def __init__(self, config, tokenizer, workspace):
        self.config, self.tokenizer = config, tokenizer
        self.assets = {"punkt_tab.zip": tokenizer.asset_record}
        self.connection = sqlite3.connect(workspace / "ngrams.sqlite")
        self.connection.executescript("""
            PRAGMA cache_size=-8192;
            CREATE TABLE grams(gram TEXT, doc INTEGER, PRIMARY KEY(gram,doc)) WITHOUT ROWID;
            CREATE TEMP TABLE current(gram TEXT PRIMARY KEY) WITHOUT ROWID;
        """)
        self.parents, self.sizes, self.ids, self.edges = [], [], [], []
        self.exact = {}

    def close(self):
        self.connection.close()

    def find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, first, second, score, kind):
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parents[max(a, b)] = min(a, b)
            self.edges.append((first, second, score, kind))

    def add(self, document):
        index = len(self.ids)
        if index >= self.config["max_documents"]:
            raise ValueError("Dedup max_documents exceeded; exact comparison is intended for small samples")
        self.ids.append(document["document_id"])
        self.parents.append(index)
        normalized = simplify_text(document["text"])
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).digest()
        if fingerprint in self.exact:
            self.union(index, self.exact[fingerprint], 1.0, "normalized_exact")
        else:
            self.exact[fingerprint] = index
        words, n = self.tokenizer.word_tokenize(normalized), self.config["ngram_size"]
        # Store actual n-gram strings: no probabilistic shingle-hash collisions.
        grams = {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
        size = len(grams)
        self.sizes.append(size)
        self.connection.execute("DELETE FROM current")
        self.connection.executemany("INSERT INTO current VALUES (?)", ((gram,) for gram in sorted(grams)))
        candidates = self.connection.execute("""
            SELECT grams.doc, COUNT(*) FROM current CROSS JOIN grams USING(gram)
            GROUP BY grams.doc ORDER BY grams.doc
        """)
        for other, intersection in candidates:
            union = size + self.sizes[other] - intersection
            score = intersection / union
            if score >= self.config["similarity_threshold"]:
                self.union(index, other, score, "ngram_jaccard")
        self.connection.executemany("INSERT INTO grams VALUES (?,?)", ((gram, index) for gram in sorted(grams)))
        self.connection.commit()

    def decisions(self):
        # A spanning tree of witnessed edges explains transitive removals without
        # claiming every removed row meets the threshold against the survivor.
        adjacency = [[] for _ in self.ids]
        for a, b, score, kind in self.edges:
            adjacency[a].append((b, score, kind))
            adjacency[b].append((a, score, kind))
        decisions = {}
        for root in range(len(self.ids)):
            if self.find(root) != root:
                continue
            decisions[self.ids[root]] = (True, None, {"representative_id": self.ids[root]})
            stack, seen = [root], {root}
            while stack:
                current = stack.pop()
                for neighbor, score, kind in sorted(adjacency[current]):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    decisions[self.ids[neighbor]] = (False, "dedup_" + kind, {
                        "representative_id": self.ids[root], "matched_document_id": self.ids[current],
                        "similarity": score, "threshold": self.config["similarity_threshold"],
                        "match_type": kind, "transitive_cluster": current != root})
                    stack.append(neighbor)
        return decisions
