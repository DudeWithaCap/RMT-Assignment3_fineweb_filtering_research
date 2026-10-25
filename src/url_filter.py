"""FineWeb URL rules with pinned full lists and a disk-backed lookup table."""

import re
import sqlite3
import tarfile

from .filter_assets import checked_asset

NORMALIZER = re.compile(r"[^a-zA-Z0-9]+")


class URLFilter:
    def __init__(self, config, assets_directory, workspace):
        import tldextract

        self.config = config
        self.assets = {}
        paths = {}
        for name in ("url_filterblacklists.tar.gz", "banned_words.txt", "banned_subwords.txt", "soft_banned_words.txt"):
            paths[name], self.assets[name] = checked_asset(assets_directory, name)
        # Freeze suffix interpretation to the installed package snapshot, no network/cache.
        self.extract = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())
        self.connection = sqlite3.connect(workspace / "url_lists.sqlite")
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.execute("CREATE TABLE blocked(kind TEXT, value TEXT, PRIMARY KEY(kind,value)) WITHOUT ROWID")
        try:
            with tarfile.open(paths["url_filterblacklists.tar.gz"], "r:gz") as archive:
                for member, kind in (("adult/domains", "domain"), ("adult/urls", "url")):
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Missing blocklist member: {member}")
                    with stream:
                        # Read archive members as data, never extract paths or links.
                        batch = []
                        for line in stream:
                            value = line.decode("utf-8").strip()
                            if value and not value.startswith("#"):
                                batch.append((kind, value))
                            if len(batch) >= 5000:
                                self.connection.executemany("INSERT OR IGNORE INTO blocked VALUES (?,?)", batch)
                                batch.clear()
                        self.connection.executemany("INSERT OR IGNORE INTO blocked VALUES (?,?)", batch)
            for kind, key in (("domain", "extra_domains"), ("url", "extra_urls")):
                self.connection.executemany("INSERT OR IGNORE INTO blocked VALUES (?,?)",
                                            ((kind, value) for value in config[key]))
            self.connection.commit()
        except Exception:
            self.close()
            raise
        self.words = {}
        for key in ("banned_words", "banned_subwords", "soft_banned_words"):
            self.words[key] = sorted({NORMALIZER.sub("", line).lower()
                                     for line in paths[key + ".txt"].read_text(encoding="utf-8").splitlines()
                                     if line and not line.startswith("#")})

    def close(self):
        self.connection.close()

    def _blocked(self, kind, value):
        return bool(value) and self.connection.execute(
            "SELECT 1 FROM blocked WHERE kind=? AND value=?", (kind, value)).fetchone() is not None

    def evaluate(self, document):
        url = document["url"]
        extracted = self.extract(url)
        for value, reason, kind in ((extracted.top_domain_under_public_suffix, "domain", "domain"),
                                    (extracted.fqdn, "subdomain", "domain"), (url, "url", "url")):
            if self._blocked(kind, value):
                return False, "url_" + reason, {"matched": value}
        # Retain the historical code's case-sensitive token matching. Subwords are lowercase.
        tokens = set(NORMALIZER.split(url))
        hard = sorted(tokens.intersection(self.words["banned_words"]))
        if hard:
            return False, "url_hard_blacklisted", {"matched": hard}
        soft = sorted(tokens.intersection(self.words["soft_banned_words"]))
        if len(soft) >= self.config["soft_word_threshold"]:
            return False, "url_soft_blacklisted", {"matched": soft}
        normalized = NORMALIZER.sub("", url).lower()
        matches = [word for word in self.words["banned_subwords"] if word in normalized]
        if matches:
            return False, "url_blacklisted_subword", {"matched": matches}
        return True, None, {}
