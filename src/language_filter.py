"""Pretrained fastText top-one English language gate."""

from .filter_assets import checked_asset
import shutil
import tempfile
from pathlib import Path


class LanguageFilter:
    def __init__(self, config, assets_directory, model=None):
        self.threshold = config["english_confidence_threshold"]
        self.assets = {}
        if model is None:
            import fasttext

            path, self.assets[config["model_file"]] = checked_asset(assets_directory, config["model_file"])
            if str(path).isascii():
                model = fasttext.load_model(str(path))
            else:
                # fastText's Windows C++ loader cannot open non-ASCII filenames.
                # Copy the same verified bytes; inference remains in memory.
                with tempfile.TemporaryDirectory(prefix="rmt_fasttext_") as directory:
                    copy = Path(directory) / config["model_file"]
                    if not str(copy).isascii():
                        raise ValueError("fastText on Windows requires an ASCII model path; configure TEMP to an ASCII directory")
                    shutil.copyfile(path, copy)
                    model = fasttext.load_model(str(copy))
        self.model = model

    def evaluate(self, document):
        # Original DataTrove removes LF rather than replacing it with a space.
        text = document["text"].replace("\n", "")
        if not text.strip():
            return False, "language_empty_text", {"language": None, "confidence": None}
        labels, scores = self.model.predict(text, k=1)
        language, score = labels[0].removeprefix("__label__"), float(scores[0])
        details = {"language": language, "confidence": score, "threshold": self.threshold}
        if language != "en":
            return False, "language_non_english", details
        if score <= self.threshold:
            return False, "language_low_confidence", details
        return True, None, details
