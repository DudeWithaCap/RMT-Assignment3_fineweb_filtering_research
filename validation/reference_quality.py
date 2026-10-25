"""Validation-only harness for pinned upstream quality sources.

Original rule bodies/defaults execute unchanged. DataTrove framework plumbing is
replaced by minimal adapters, sharing the project's installed NLTK tokenizer.
This checks rule parity, not historical tokenizer or HTML extraction parity.
"""

import ast
from collections import Counter
import heapq
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from src.filter_assets import DATATROVE_REVISION, sha256_file

REFERENCE_FILES = (
    "gopher_repetition_filter.py", "gopher_quality_filter.py",
    "c4_filters.py", "fineweb_quality_filter.py", "text.py", "fineweb.py", "language_filter.py",
)


class FrameworkAdapter:
    def __init__(self, *args, **kwargs):
        pass

    def stat_update(self, *args, **kwargs):
        pass


def build_reference(directory, tokenizer):
    lock = json.loads((Path(__file__).parent / "reference_sources.json").read_text(encoding="utf-8"))
    if lock["revision"] != DATATROVE_REVISION:
        raise ValueError("Reference revision mismatch")
    for filename in REFERENCE_FILES:
        if sha256_file(directory / filename) != lock["sha256"][filename]:
            raise ValueError(f"Changed upstream reference: {filename}")
    namespace = {"BaseFilter": FrameworkAdapter, "DiskWriter": object, "Document": object,
                 "Languages": SimpleNamespace(english="eng"), "load_word_tokenizer": lambda language: tokenizer,
                 "np": np, "re": re, "Counter": Counter, "heapq": heapq}
    punctuation = ast.parse((directory / "text.py").read_text(encoding="utf-8"))
    assignments = [node for node in punctuation.body if isinstance(node, ast.Assign)
                   and any(getattr(target, "id", "") in ("PUNCTUATION", "PUNCTUATION_SET") for target in node.targets)]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "reference_punctuation", "exec"), namespace)
    classes = ("GopherRepetitionFilter", "GopherQualityFilter", "C4QualityFilter", "FineWebQualityFilter")
    for filename, wanted in zip(REFERENCE_FILES[:4], classes):
        tree = ast.parse((directory / filename).read_text(encoding="utf-8"))
        statements = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))
                      or isinstance(node, ast.ClassDef) and node.name == wanted]
        exec(compile(ast.Module(body=statements, type_ignores=[]), filename, "exec"), namespace)
    filters = [namespace[name](**({"filter_no_terminal_punct": False} if name == "C4QualityFilter" else {}))
               for name in classes]
    return filters, namespace


def reference_decision(filters, text):
    document = SimpleNamespace(text=text)
    for stage in filters:
        result = stage.filter(document)
        kept, reason = result if isinstance(result, tuple) else (result, None)
        if not kept:
            return False, type(stage).__name__, reason
    return True, None, None
