from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams

from lm4bld.models.nl import NGramModel
from lm4bld.models.pom import CType
from lm4bld.models.pom import PomModel

class EnsembleModel(PomModel):
    def __init__(self, order, tokenizer, prefix, tokenprefix, ignore_syntax):
        super().__init__(order, tokenizer, prefix, tokenprefix, ignore_syntax)
        self.lm = NGramModel(order, tokenizer, prefix, tokenprefix, ignore_syntax)
        self.my_tokenizer = tokenizer(None, prefix, tokenprefix, ignore_syntax, True, True)

    def payload_sents(self):
        mymap = self.map.flattened_view(CType.TAGCONTENTLOC)
        mysents = list()

        for location in mymap:
            for payload in mymap[location]:
                tokens = location.split("/")
                tokens += self.tokenize(payload)
                mysents.append(tokens)

        return mysents

    def fit(self, flist):
        super().fit(flist)
        self.lm.fit(self.payload_sents(), False)

    def tokenize(self, term):
        preprocessed_term = self.my_tokenizer.preprocess_strings(term)
        tokens = self.my_tokenizer.tokenize_string(preprocessed_term)
        clean_tokens = tokens

        if (self.ignore_syntax):
            to_clean = list()
            to_clean.append(tokens)
            clean_tokens = self.my_tokenizer.remove_syntax(to_clean)[0]

        return clean_tokens

    def grammify(self, context, term):
        tokens = context.split("/")
        tokens += self.tokenize(term)
        padded_tokens = list(pad_both_ends(tokens, n=self.order))
        return list(everygrams(padded_tokens, max_len=self.order))

    def payloadGramscore(self, context, payload):
        gramScore = 0
        ngrams = self.grammify(context, payload)

        # Ripped from nltk.lm.api
        for ngram in ngrams:
            score = self.lm.model.logscore(ngram[-1], ngram[:-1])
            gramScore += score

        return gramScore/len(ngrams) if len(ngrams) > 0 else None

    def logscore(self, ctype, context, term):
        return self.payloadGramscore(context, term) if (ctype is CType.TAGCONTENTLOC or ctype is CType.TAGCONTENT) else super().logscore(ctype, context, term)

    def unk_tokens(self, ctype, context, term):
        rtn = None

        if (ctype is CType.TAGCONTENTLOC or ctype is CType.TAGCONTENT):
            ngrams = self.grammify(context, term)
            unk_count = 0
            for gram in ngrams:
                gram_or_unk = self.lm.model.vocab.lookup(gram)
                if (gram_or_unk[0] == "<UNK>"):
                    unk_count += 1

            rtn = [unk_count, len(ngrams)]
        else:
            rtn = super().unk_tokens(ctype, context, term)

        return rtn

    def processGramForNextTokenExp(self, gram, nCandidates):
        ctype = gram[0]
        context = gram[1]
        token = gram[2]

        correct = {}
        incorrect = {}

        if (ctype is CType.TAGCONTENTLOC or ctype is CType.TAGCONTENT):
            ngrams = self.grammify(context, token)
            for ngram in ngrams:
                to_guess = ngram[-1]
                if self.my_tokenizer.is_syntax(to_guess) or to_guess.isnumeric() or to_guess.isspace():
                    continue

                guesses = self.lm.guessNextToken(ngram[:-1], nCandidates)
                if (to_guess in guesses):
                    if len(to_guess) not in correct:
                        correct[len(to_guess)] = list()

                    correct[len(to_guess)].append(to_guess)

                else:
                    if len(to_guess) not in incorrect:
                        incorrect[len(to_guess)] = list()

                    incorrect[len(to_guess)].append(to_guess)

        else:
            correct, incorrect = super().processGramForNextTokenExp(gram, nCandidates)

        return correct, incorrect
