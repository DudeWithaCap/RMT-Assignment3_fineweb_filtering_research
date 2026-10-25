"""Document decisions from FineWeb's Gopher, C4 and custom quality rules.

C4's cleaned working text is used for downstream checks within this stage only.
The runner writes the original document, preserving the shared baseline text.
"""

from collections import Counter
import re

from .filter_text import PUNCTUATION_SET, duplicate_counts

POLICY = ("terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies", "use cookies")
CITATIONS = re.compile(r"\[\d*]|\[edit]|\[citation needed]")


class QualityFilter:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.assets = {"punkt_tab.zip": tokenizer.asset_record}

    def evaluate(self, document):
        config, text = self.config, document["text"]
        metrics = {}

        def check(name, value, threshold, operator=">"):
            metrics[name] = {"value": value, "threshold": threshold, "reject_when": operator}
            return {">": value > threshold, "<": value < threshold,
                    ">=": value >= threshold, "<=": value <= threshold}[operator]

        def reject(reason):
            return False, reason, metrics

        if not text.strip():
            return reject("quality_empty_text")

        # Gopher repetition: duplicate paragraphs, lines, then repeated n-grams.
        for parts, element_key, char_key in (
                (re.split(r"\n{2,}", text.strip()), "dup_para_frac", "dup_para_char_frac"),
                (re.split(r"\n+", text), "dup_line_frac", "dup_line_char_frac")):
            repeated, characters = duplicate_counts(parts)
            if check(element_key, repeated / len(parts), config[element_key]):
                return reject("quality_" + element_key)
            if check(char_key, characters / len(text), config[char_key]):
                return reject("quality_" + char_key)
        words = self.tokenizer.word_tokenize(text)
        if not words:
            return reject("quality_no_tokens")
        for number, threshold in sorted(config["top_ngram_thresholds"].items(), key=lambda x: int(x[0])):
            n = int(number)
            counts = Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
            if counts:
                # Counter preserves first-occurrence ties, as in the reference code.
                gram, frequency = counts.most_common(1)[0]
                name = f"top_{n}_gram"
                if check(name, len(gram) * frequency / len(text), threshold):
                    return reject("quality_" + name)
        for number, threshold in sorted(config["duplicate_ngram_thresholds"].items(), key=lambda x: int(x[0])):
            n, index, repeated, seen = int(number), 0, 0, set()
            while index <= len(words) - n:
                gram = "".join(words[index:index + n])
                if gram in seen:
                    repeated += len(gram)
                    index += n  # avoid double-counting overlapping repeated spans
                else:
                    seen.add(gram)
                    index += 1
            name = f"duplicated_{n}_ngrams"
            if check(name, repeated / len(text), threshold):
                return reject("quality_" + name)

        # Gopher quality. Counts exclude punctuation-only tokens for word length.
        content_words = [word for word in words if any(c not in PUNCTUATION_SET for c in word)]
        if check("min_doc_words", len(content_words), config["min_doc_words"], "<"):
            return reject("quality_short_doc")
        if check("max_doc_words", len(content_words), config["max_doc_words"]):
            return reject("quality_long_doc")
        average = sum(map(len, content_words)) / max(1, len(content_words))
        if check("min_avg_word_length", average, config["min_avg_word_length"], "<"):
            return reject("quality_short_words")
        if check("max_avg_word_length", average, config["max_avg_word_length"]):
            return reject("quality_long_words")
        for name, count in (("hash_symbol_ratio", text.count("#")),
                            ("ellipsis_symbol_ratio", text.count("...") + text.count("…"))):
            if check(name, count / len(words), config["max_symbol_word_ratio"]):
                return reject("quality_" + name)
        lines = text.splitlines()
        if check("bullet_lines_ratio", sum(line.lstrip().startswith(("•", "-")) for line in lines) / len(lines),
                 config["max_bullet_lines_ratio"]):
            return reject("quality_bullet_lines")
        if check("ellipsis_lines_ratio", sum(line.rstrip().endswith(("...", "…")) for line in lines) / len(lines),
                 config["max_ellipsis_lines_ratio"]):
            return reject("quality_ellipsis_lines")
        alpha_ratio = sum(any(c.isalpha() for c in word) for word in words) / len(words)
        if check("alpha_word_ratio", alpha_ratio, config["min_alpha_word_ratio"], "<"):
            return reject("quality_non_alpha_words")
        stop_count = sum(word in config["stop_words"] for word in words)
        if check("stop_word_occurrences", stop_count, config["min_stop_words"], "<"):
            return reject("quality_insufficient_stop_words")

        # C4: terminal punctuation rule disabled, as in FineWeb's example pipeline.
        kept, line_reasons, sentences = [], Counter(), 0
        for line in lines:
            line = line.strip()
            whitespace_words = line.split()
            if any(len(word) > config["c4_max_word_length"] for word in whitespace_words):
                line_reasons["long_word"] += 1
                continue
            if config["c4_remove_citations"]:
                line = CITATIONS.sub("", line)
            if len(whitespace_words) < config["c4_min_words_per_line"]:
                line_reasons["short_line"] += 1
                continue
            lowered = line.lower()
            if config["c4_filter_lorem_ipsum"] and "lorem ipsum" in lowered:
                return reject("quality_c4_lorem_ipsum")
            if config["c4_filter_javascript"] and "javascript" in lowered:
                line_reasons["javascript"] += 1
                continue
            if config["c4_filter_curly_bracket"] and "{" in line:
                return reject("quality_c4_curly_bracket")
            if config["c4_filter_policy"] and any(phrase in lowered for phrase in POLICY):
                line_reasons["policy"] += 1
                continue
            sentences += len(self.tokenizer.sent_tokenize(line))
            kept.append(line)
        metrics["c4_removed_lines"] = dict(line_reasons)
        if check("c4_sentence_count", sentences, config["c4_min_sentences"], "<"):
            return reject("quality_c4_too_few_sentences")
        working_text = "\n".join(kept).strip()
        metrics["c4_working_characters"] = len(working_text)
        if not working_text:
            return reject("quality_c4_empty_working_text")

        # FineWeb-specific document rules, applied to C4's working text.
        lines = working_text.split("\n")
        punctuation_ratio = sum(line.endswith((".", "'", '"', "!", "?")) for line in lines) / len(lines)
        if check("line_punct_ratio", punctuation_ratio, config["line_punct_threshold"], "<="):
            return reject("quality_line_punct_ratio")
        short_ratio = sum(len(line) <= config["short_line_length"] for line in lines) / len(lines)
        if check("short_line_ratio", short_ratio, config["short_line_threshold"], ">="):
            return reject("quality_short_line_ratio")
        nonempty = [line for line in lines if line.strip()]
        ratio = duplicate_counts(nonempty)[1] / max(1, len(working_text.replace("\n", "")))
        if check("char_dup_ratio", ratio, config["char_duplicates_ratio"], ">="):
            return reject("quality_char_dup_ratio")
        token_count = len(self.tokenizer.word_tokenize(working_text))
        ratio = working_text.count("\n") / max(1, token_count)
        if check("newline_word_ratio", ratio, config["newline_word_ratio"]):
            return reject("quality_list_ratio")
        return True, None, metrics
