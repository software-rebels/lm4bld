from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams

from lm4bld.models.nl import NGramModel
from lm4bld.models.pom import CType
from lm4bld.models.pom import PomModel
from lm4bld.models.tokenize import PomTokenizer

class EnsembleModel(PomModel):
    def __init__(self, order, ignore_syntax):
        super().__init__(order, ignore_syntax)
        self.lm = NGramModel(order, ignore_syntax)
        self.tokenizer = PomTokenizer(None, None, None, True, True)
        self.ignore_syntax = ignore_syntax

    def get_sents(self, flist):
        sents = list()

        for f in flist:
            t = PomTokenizer(f, None, None, True, True)
            sents += t.tokens_to_sents()

        return sents

    def fit(self, flist):
        super().fit(flist)
        self.lm.fit(self.get_sents(flist))

    def grammify(self, term):
        preprocessed_term = self.tokenizer.preprocess_strings(term)
        tokens = self.tokenizer.tokenize_string(preprocessed_term)
        clean_tokens = self.tokenizer.remove_syntax(tokens) if (self.ignore_syntax) else tokens
        padded_tokens = list(pad_both_ends(clean_tokens, n=self.order))
        return list(everygrams(padded_tokens, max_len=self.order))

    def payloadGramscore(self, context, term):
        gramScore = 0
        ngrams = self.grammify(term)

        # Ripped from nltk.lm.api
        for ngram in ngrams:
            gramScore += self.lm.model.logscore(ngram[-1], ngram[:-1])

        return gramScore/len(ngrams) if len(ngrams) > 0 else 0

    def logscore(self, context, term, ctype):
        gramScore = None

        match ctype:
            case CType.TAGCONTENTLOC:
                gramScore = self.payloadGramscore(context, term)
            case _:
                gramScore = super().logscore(context, term, ctype)

        return gramScore

    def unk_tokens(self, context, term, ctype):
        rtn = None

        match ctype:
            case CType.TAGCONTENTLOC:
                ngrams = self.grammify(term)
                unk_count = 0

                for gram in ngrams:
                    gram_or_unk = self.lm.model.vocab.lookup(gram)
                    if (gram_or_unk[0] == "<UNK>"):
                        unk_count += 1

                rtn = [unk_count, len(ngrams)]
            case _:
                rtn = super().unk_tokens(context, term, ctype)

        return rtn
