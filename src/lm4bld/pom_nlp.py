import itertools
import random
import re

from nltk.tokenize import RegexpTokenizer
from nltk.lm import KneserNeyInterpolated
from nltk.lm import Lidstone
from nltk.lm import AbsoluteDiscountingInterpolated
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import everygrams
from nltk.util import ngrams 

class PomTokenizer:
    def __init__(self, filename):
        self.fhandle = open(filename, 'r')

    def removeComments(self, strdata):
        pattern = re.compile(r"\<\!\-\-[\s\S]*?\-\-\>")

        return re.sub(pattern, "", strdata)

    def tokenize(self):
        strdata = self.removeComments(self.fhandle.read())

        tokenizer = RegexpTokenizer(
            '[\"\'][\s\S]*?[\"\']|\n|\<\?|\?\>|\<\/|\/\>|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')

        return tokenizer.tokenize(strdata)

    # Not sure if we should do this for Maven
    def normalize(self, sent):
        return sent.lower()

    def sentence_tokenize(self):
        sents = list()
        toks = self.tokenize()

        sent = list()
        for tok in toks:
            if tok == "\n":
                sents.append(sent)
                sent = list()
            else:
                sent.append(self.normalize(tok))

        return sents

class PomNGramModel:
    def __init__(self, order):
        self.order = order
        #self.model = Lidstone(0.01, self.order)
        empty_vocab = Vocabulary(unk_cutoff=1)
        self.model = AbsoluteDiscountingInterpolated(self.order, vocabulary=empty_vocab)

    def fit(self, train_sents):
        train_corp, vocab = padded_everygram_pipeline(self.order, train_sents)
        self.model.fit(train_corp, vocab)

    def crossEntropy(self, ngrams):
        return self.model.entropy(ngrams)

    def unkRate(self, sents):
        unk_count = 0
        unigrams = set()
        for sent in sents:
            unigrams.update(list(ngrams(sent, 1)))

        for gram in unigrams:
            gram_or_unk = self.model.vocab.lookup(gram)
            if (gram_or_unk[0] == "<UNK>"):
                print(gram)
                unk_count += 1

        print(unk_count)
        print(len(unigrams))

        return unk_count / len(unigrams)

class PomNLPExperiment:
    def __init__(self, pomlistfile, order):
        self.order = order

        fhandle = open(pomlistfile, 'r')
        listOfPoms = fhandle.readlines()

        self.allSents = list()
        for file in listOfPoms:
            tokenizer = PomTokenizer(file.strip())
            self.allSents += tokenizer.sentence_tokenize()

        random.seed(666)

    def getFolds(self, nfolds):
        foldSents = list()

        for fold in range(nfolds):
            foldSents.append(list())

        myList = self.allSents
        random.shuffle(myList)

        cntr = 0
        for sent in myList:
            foldSents[cntr % nfolds].append(sent)
            cntr += 1

        return foldSents

    def testModel(self, fitter, sents):
        ngrams = list()

        for sent in sents:
            paddedTokens = list(pad_both_ends(sent, n=self.order))
            ngrams += list(everygrams(paddedTokens, max_len=self.order))

        #return fitter.crossEntropy(ngrams)
        return fitter.unkRate(sents)

    def nFoldValidation(self, nfolds=10, niter=1):
        resultsList = list()

        for i in range(niter):
            myFolds = self.getFolds(nfolds)

            for testIdx in range(nfolds):
                print("Fold %i (iteration %i)" % (testIdx, i))
                testCorpus = myFolds[testIdx]

                trainCorpus = list()
                for trainIdx in range(nfolds):
                    if trainIdx != testIdx:
                        trainCorpus += myFolds[trainIdx]

                # Train model on training corpus
                fitter = PomNGramModel(self.order)
                fitter.fit(trainCorpus)

                # Test it on testing corpus
                resultsList.append(self.testModel(fitter, testCorpus))

        # Return results list
        return resultsList

class PomNLPMultirunExperiment:
    def __init__(self, pomlistfile, minorder, maxorder):
        self.pomlistfile = pomlistfile
        self.minorder = minorder
        self.maxorder = maxorder

    def perform(self, nfolds=10, niter=1):
        results = {}

        for order in range(self.minorder, (self.maxorder+1)):
            exp = PomNLPExperiment(self.pomlistfile, order)
            results[order] = exp.nFoldValidation(nfolds, niter)

        return results
