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

    def fit(self, trainCorpus, filelevel=True):
        train_sents = self.load_sents(trainCorpus) if filelevel else trainCorpus
        train_corp, vocab = padded_everygram_pipeline(self.order, train_sents)
        self.model.fit(train_corp, vocab)

    def crossEntropy(self, flist):
        ngrams = self.grammify(self.load_sents(flist))
        return self.model.entropy(ngrams)

    def unkRate(self, flist):
        ngrams = self.grammify(self.load_sents(flist))
        unk_count = 0

        for gram in ngrams:
            gram_or_unk = self.model.vocab.lookup(gram)
            if (gram_or_unk[0] == "<UNK>"):
                #print(gram)
                unk_count += 1

        return unk_count / len(ngrams)

    def guessNextTokens(self, testCorpus, nCandidates, filelevel=True):
        correct = {}
        incorrect = {}

        sents = self.load_sents(testCorpus) if filelevel else testCorpus

        for sent in sents:
            context = ("<s>",) * (self.order-1) # Context is initialized with padding up the order
            for token in sent:
                guesses = self.guessNextToken(context, nCandidates)
                if (token in guesses):
                    if len(token) not in correct:
                        correct[len(token)] = list()

                    correct[len(token)].append(token)

                else:
                    if len(token) not in incorrect:
                        incorrect[len(token)] = list()

                    incorrect[len(token)].append(token)

                context = context[1:] + (token,)

        rtn = {}
        allkeys = set(list(correct.keys()) + list(incorrect.keys()))
        for token_len in allkeys:
            n_correct = len(correct[token_len]) if token_len in correct else 0
            n_incorrect = len(incorrect[token_len]) if token_len in incorrect else 0
            rtn[token_len] = [n_correct, n_incorrect]

        return rtn

    def guessNextToken(self, context, nCandidates):
        sorted_sample = sorted(self.model.context_counts(context))
        return sorted_sample[0:nCandidates]
