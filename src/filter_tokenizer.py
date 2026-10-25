"""NLTK English tokenizer backed by checked, local, non-pickle tables."""

import io
import zipfile

from .filter_assets import checked_asset


class EnglishTokenizer:
    def __init__(self, assets_directory):
        from nltk.tabdata import PunktDecoder
        from nltk.tokenize import NLTKWordTokenizer
        from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

        path, self.asset_record = checked_asset(assets_directory, "punkt_tab.zip")
        decoder = PunktDecoder()
        parameters = PunktParameters()
        with zipfile.ZipFile(path) as archive:
            def read(name):
                data = archive.read("punkt_tab/english/" + name)
                if len(data) > 2 * 1024 * 1024:
                    raise ValueError("Unexpectedly large tokenizer table")
                return io.StringIO(data.decode("utf-8"))

            parameters.collocations = set(decoder.tab2tups(read("collocations.tab")))
            parameters.sent_starters = decoder.txt2set(read("sent_starters.txt"))
            parameters.abbrev_types = decoder.txt2set(read("abbrev_types.txt"))
            parameters.ortho_context = decoder.tab2intdict(read("ortho_context.tab"))
        self.sentences = PunktSentenceTokenizer(parameters)
        self.words = NLTKWordTokenizer()

    def sent_tokenize(self, text):
        return [sentence.strip() for sentence in self.sentences.tokenize(text) if sentence.strip()]

    def word_tokenize(self, text):
        return [word.strip() for sentence in self.sent_tokenize(text)
                for word in self.words.tokenize(sentence) if word.strip()]
