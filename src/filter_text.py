"""Text normalization compatible with the pinned DataTrove defaults."""

import re
import unicodedata

PUNCTUATION = "!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»"
PUNCTUATION += "".join(chr(x) for a, b in ((0, 9), (11, 13), (13, 32), (127, 160)) for x in range(a, b))
PUNCTUATION_SET = set(PUNCTUATION)


def simplify_text(text):
    text = re.sub(r"\s+", " ", text.lower().strip())
    text = text.translate(str.maketrans("", "", PUNCTUATION))
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"\d+", "0", text).strip()


def duplicate_counts(parts):
    seen, count, characters = set(), 0, 0
    for part in parts:
        if part in seen:
            count += 1
            characters += len(part)
        seen.add(part)
    return count, characters
