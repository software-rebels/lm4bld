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

    def preprocess_strings(self, strdata):
        # Comments
        comment_cleanup = re.compile(r"\<\!\-\-[\s\S]*?\-\-\>")
        strdata = re.sub(comment_cleanup, "", strdata)

        # Version strings
        version_cleanup = re.compile(r"\d+(?:\.[v\d]+)+(?:[-\w]*)?")
        strdata =re.sub(version_cleanup, "<VERSNUM>", strdata)

        # Path strings
        path_cleanup = re.compile(r"\w+(?:\/[\w\.\*\-]+)+")
        strdata = re.sub(path_cleanup, "<PATHSTR>", strdata)

        return strdata

    def tokenize(self):
        strdata = self.preprocess_strings(self.fhandle.read())

        tokenizer = RegexpTokenizer(
            #'[\"\'][\s\S]*?[\"\']|\d+(?:\.\d+)+(?:[-\w]*)?|\w+(?:\/[\w\.\*\-]+)+|\w+(?:\.\w+)+|\n|\<\?|\?\>|\<\/|\/\>|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')
            '[\"\']|\n|\<\?|\?\>|\<\/|\/\>|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')

        return tokenizer.tokenize(strdata)

    # Not sure if we should do this for Maven
    def normalize(self, tok):
        return tok.lower()

    def sentence_tokenize(self):
        sents = list()
        toks = self.tokenize()

        
       # if tok.startswith("\"") or tok.startswith("\'"):
       #     tok = "<STRLIT>"
       # elif re.match(r"\d+(?:\.\d+)+(?:[-\w]*)?", tok):
       #     tok = "<VERSNUM>"
       # elif re.match(r"\w+(?:\/[\w\.\*\-]+)+", tok):
       #     tok = "<PATHLIT>"
       # elif re.match(r"\w+(?:\.\w+)+", tok):
       #     tok = "<URI>"

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
        self.model = Lidstone(0.00017, self.order)
        empty_vocab = Vocabulary(unk_cutoff=1)
        #self.model = AbsoluteDiscountingInterpolated(self.order, vocabulary=empty_vocab)

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

class PomNLPExperiment:
    def __init__(self, project, pomlistfile, order):
        self.project = project
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

        return fitter.unkRate(ngrams), fitter.crossEntropy(ngrams)

    def nFoldValidation(self, nfolds=10, niter=1):
        for iteration in range(niter):
            myFolds = self.getFolds(nfolds)

            for fold in range(nfolds):
                testCorpus = myFolds[fold]

                trainCorpus = list()
                for trainIdx in range(nfolds):
                    if trainIdx != fold:
                        trainCorpus += myFolds[trainIdx]

                # Train model on training corpus
                fitter = PomNGramModel(self.order)
                fitter.fit(trainCorpus)

                # Test it on testing corpus
                unk_rate, entropy = self.testModel(fitter, testCorpus)
                print(f'{self.project},unk_rate,{unk_rate},{self.order},{fold},{iteration}')
                print(f'{self.project},entropy,{entropy},{self.order},{fold},{iteration}')

class PomNLPMultirunExperiment:
    def __init__(self, project, pomlistfile, minorder, maxorder):
        self.project = project
        self.pomlistfile = pomlistfile
        self.minorder = minorder
        self.maxorder = maxorder

    def perform(self, nfolds=10, niter=1):
        for order in range(self.minorder, (self.maxorder+1)):
            exp = PomNLPExperiment(self.project, self.pomlistfile, order)
            exp.nFoldValidation(nfolds, niter)
