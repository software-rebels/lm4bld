from nltk.lm import Lidstone

from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.vocabulary import Vocabulary
from nltk.util import everygrams

from lm4bld.models.api import Model

class NGramModel(Model):
    def __init__(self, order, tokenizer, prefix, tokenprefix, ignore_syntax):
        super().__init__(order, tokenizer, prefix, tokenprefix, ignore_syntax)
        self.model = Lidstone(0.00017, self.order)
        empty_vocab = Vocabulary(unk_cutoff=1)

    def load_sents(self, flist):
        sents = list()

        for f in flist:
            t = self.tokenizer(f, self.prefix, self.tokenprefix,
                               self.ignore_syntax)
            sents += t.load_tokens()

        return sents

    def grammify(self, test_sents):
        ngrams = list()

        for sent in test_sents:
            paddedTokens = list(pad_both_ends(sent, n=self.order))
            ngrams += list(everygrams(paddedTokens, max_len=self.order))

        return ngrams

    def fit(self, trainCorpus, filelevel):
        train_sents = self.load_sents(trainCorpus) if filelevel else trainCorpus
        train_corp, vocab = padded_everygram_pipeline(self.order, train_sents)
        self.model.fit(train_corp, vocab)

    def crossEntropy(self, indata, filelevel):
        ngrams = self.grammify(self.load_sents(indata)) if filelevel else indata
        return self.model.entropy(ngrams)

    def unkRate(self, indata, filelevel):
        ngrams = self.grammify(self.load_sents(indata)) if filelevel else indata
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
