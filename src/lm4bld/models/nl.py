from nltk.lm import AbsoluteDiscountingInterpolated
from nltk.lm import KneserNeyInterpolated
from nltk.lm import Lidstone

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.vocabulary import Vocabulary

class NGramModel:
    def __init__(self, order, ignore_syntax):
        self.order = order
        self.model = Lidstone(0.00017, self.order)
        empty_vocab = Vocabulary(unk_cutoff=1)
        self.ignore_syntax = ignore_syntax

    def fit(self, train_sents):
        train_corp, vocab = padded_everygram_pipeline(self.order, train_sents)
        self.model.fit(train_corp, vocab)

    def crossEntropy(self, ngrams):
        return self.model.entropy(ngrams)

    def unkRate(self, ngrams):
        unk_count = 0

        for gram in ngrams:
            gram_or_unk = self.model.vocab.lookup(gram)
            if (gram_or_unk[0] == "<UNK>"):
                #print(gram)
                unk_count += 1

        return unk_count / len(ngrams)

    def guessNextToken(self, context, nCandidates):
        sorted_sample = sorted(self.model.context_counts(context))
        return sorted_sample[0:nCandidates]
