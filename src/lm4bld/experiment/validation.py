#import itertools
from abc import ABCMeta, abstractmethod
import os
import random

from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams

from lm4bld.nlp.model import NGramModel

class NLPValidator(metaclass=ABCMeta):
    def __init__(self, project, listfile, tokenizer, order, versions, paths):
        self.project = project
        self.order = order
        self.versions = versions
        self.paths = paths
        self.tokenizer = tokenizer

        fhandle = open(listfile, 'r')
        self.listfiles = list()
        self.allSents = list()
        for file in fhandle.readlines():
            pomfile = file.strip()
            self.listfiles.append(pomfile)
            t = tokenizer(pomfile, versions, paths)
            self.allSents += t.sentence_tokenize()

        random.seed(666)

    def getSents(self):
        return self.allSents

    def getProject(self):
        return self.project

    def trainModel(self, trainCorpus):
        fitter = NGramModel(self.order)
        fitter.fit(trainCorpus)

        return fitter

    def testModel(self, fitter, sents):
        ngrams = list()

        for sent in sents:
            paddedTokens = list(pad_both_ends(sent, n=self.order))
            ngrams += list(everygrams(paddedTokens, max_len=self.order))

        return fitter.unkRate(ngrams), fitter.crossEntropy(ngrams)

    @abstractmethod
    def validate(self, executor):
        raise NotImplementedError()
    
class CrossFoldValidator(NLPValidator):
    def __init__(self, project, listfile, tokenizer, order, versions, paths,
                 nfolds, niter):

        super().__init__(project, listfile, tokenizer, order, versions, paths)
        self.nfolds = nfolds
        self.niter = niter

    def getFolds(self):
        foldSents = list()

        for fold in range(self.nfolds):
            foldSents.append(list())

        myList = self.allSents
        random.shuffle(myList)

        cntr = 0
        for sent in myList:
            foldSents[cntr % self.nfolds].append(sent)
            cntr += 1

        return foldSents

    def nFoldJob(self, trainCorpus, testCorpus, fold, iteration):
        fitter = self.trainModel(trainCorpus)
        unk_rate, entropy = self.testModel(fitter, testCorpus)

        p = self.project
        o = self.order
        return f'{p},unk_rate,{unk_rate},{o},{fold},{iteration}{os.linesep}{p},entropy,{entropy},{o},{fold},{iteration}'

    def validate(self, executor):
        my_futures = list()

        for iteration in range(self.niter):
            myFolds = self.getFolds()

            for fold in range(self.nfolds):
                testCorpus = myFolds[fold]

                trainCorpus = list()
                for trainIdx in range(self.nfolds):
                    if trainIdx != fold:
                        trainCorpus += myFolds[trainIdx]
                
                f = executor.submit(self.nFoldJob, trainCorpus, testCorpus,
                                    fold, iteration)
                my_futures.append(f)

        return my_futures

class CrossProjectValidator(NLPValidator):
    def __init__(self, project, listfile, tokenizer, order, versions, paths):

        super().__init__(project, listfile, tokenizer, order, versions, paths)
        self.fullFit = None

    def crossProjectJob(self, otherExp):
        testCorpus = otherExp.getSents()

        if self.fullFit is None:
            self.fullFit = self.trainModel(self.allSents)

        unk_rate, entropy = self.testModel(self.fullFit, testCorpus)

        return f'{self.project},{otherExp.getProject()},{unk_rate},{entropy}'

    def setProjList(self, projlist):
        self.projlist = projlist

    def validate(self, executor):
        futures_list = list()
        trainProj = self.projlist[self.project]

        for testProjName in self.projlist:
            if (testProjName == self.project):
                continue

            f = executor.submit(trainProj.crossProjectJob,
                                self.projlist[testProjName])
            futures_list.append(f)

        return futures_list

class NextTokenValidator(NLPValidator):
    def __init__(self, project, listfile, tokenizer, order, versions, paths,
                 testSize, minCandidates, maxCandidates):
        super().__init__(project, listfile, tokenizer, order, versions, paths)
        self.testSize = testSize
        self.minCandidates = minCandidates
        self.maxCandidates = maxCandidates

    def nextTokenCorpusSplit(self):
        myList = self.listfiles
        random.shuffle(myList)

        # Testing corpus
        testCorpus = list()
        for pomfile in myList[0:self.testSize]:
            t = self.tokenizer(pomfile, self.versions, self.paths)
            testCorpus += t.sentence_tokenize()

        trainCorpus = list()
        for pomfile in myList[self.testSize:len(myList)]:
            t = self.tokenizer(pomfile, self.versions, self.paths)
            trainCorpus += t.sentence_tokenize()

        return trainCorpus, testCorpus

    def initContext(self):
        return ("<s>",) * (self.order-1)

    def guessNextTokenEvaluator(self, fitter, testCorpus, nCandidates):
        correct = {}
        incorrect = {}

        for sent in testCorpus:
            context = self.initContext()
            for token in sent:
                guesses = fitter.guessNextToken(context, nCandidates)
                if (token in guesses):
                    if len(token) not in correct:
                        correct[len(token)] = list()

                    correct[len(token)].append(token)

                else:
                    if len(token) not in incorrect:
                        incorrect[len(token)] = list()

                    incorrect[len(token)].append(token)

                context = context[1:] + (token,)

        rtnstr = ""
        for i in correct.keys():
            if rtnstr:
                rtnstr += os.linesep

            corr = len(correct[i]) if i in correct else 0
            incorr = len(incorrect[i]) if i in incorrect else 0

            rtnstr += f'{self.project},{nCandidates},{i},{corr},{incorr}'

        return rtnstr

    def validate(self, executor):
        futures_list = list()
        trainCorpus, testCorpus = self.nextTokenCorpusSplit()
        
        fitter = self.trainModel(trainCorpus)
        for nCandidates in range(self.minCandidates, (self.maxCandidates+1)):
            f = executor.submit(self.guessNextTokenEvaluator, fitter, testCorpus, nCandidates)
            futures_list.append(f)

        return futures_list
