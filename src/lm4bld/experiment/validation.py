#import itertools
from abc import ABCMeta, abstractmethod
import os
import random

from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams

from lm4bld.nlp.model import NGramModel
from lm4bld.nlp.tokenize import JavaTokenizer
from lm4bld.nlp.tokenize import PomTokenizer

class NLPValidator(metaclass=ABCMeta):
    def __init__(self, project, conf, order, listfile):
        self.project = project
        self.order = order 
        self.versions = conf.get_versions()
        self.paths = conf.get_paths()

        self.listfiles = list()
        self.allSents = list()

        # TODO: Was looped when reading through listfile
        # self.listfiles.append(fname)
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

class CrossFoldValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, order, listfile, tokenizer):
        super().__init__(project, conf, order, listfile, tokenizer)
        self.nfolds = conf.get_nfolds()
        self.niter = conf.get_niter()

    @abstractmethod
    def output_str(self, proj, unk_rate, entropy, order, fold, iteration):
        raise NotImplementedError()

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
        return self.output_str(p, unk_rate, entropy, o, fold, iteration)

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

class PomCrossFoldValidator(CrossFoldValidator):
    def __init__(self, project, conf, order):
        super().__init__(project, conf, order, conf.get_pomlist(project),
                         PomTokenizer)

    def output_str(self, proj, unk_rate, entropy, order, fold, iteration):
        unkline = f'{proj},pom,unk_rate,{unk_rate},{order},{fold},{iteration}'
        entline = f'{proj},pom,entropy,{entropy},{order},{fold},{iteration}'
        return f'{unkline}{os.linesep}{entline}'

class JavaCrossFoldValidator(CrossFoldValidator):
    def __init__(self, project, conf, order):
        super().__init__(project, conf, order, conf.get_srclist(project),
                         JavaTokenizer)

    def output_str(self, proj, unk_rate, entropy, order, fold, iteration):
        unkline = f'{proj},java,unk_rate,{unk_rate},{order},{fold},{iteration}'
        entline = f'{proj},java,entropy,{entropy},{order},{fold},{iteration}'
        return f'{unkline}{os.linesep}{entline}'

class CrossProjectValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, conf.get_crossproj_order(), listfile,
                         tokenizer)
        self.my_fit = None
        self.projects = conf.get_projects()

    @abstractmethod
    def output_str(self, train_proj, test_proj, unk_rate, entropy):
        raise NotImplementedError()

    @abstractmethod
    def get_validator(self, projname):
        raise NotImplementedError()

    def crossProjectJob(self, otherProj):
        testValidator = self.get_validator(otherProj)
        testCorpus = testValidator.getSents()

        if self.my_fit is None:
            self.my_fit = self.trainModel(self.allSents)

        unk_rate, entropy = self.testModel(self.my_fit, testCorpus)

        return self.output_str(self.project, otherProj, unk_rate, entropy)

    def validate(self, executor):
        futures_list = list()

        for testProjName in self.projects:
            if (testProjName == self.project):
                continue

            f = executor.submit(self.crossProjectJob, testProjName)
            futures_list.append(f)

        return futures_list

class PomCrossProjectValidator(CrossProjectValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)
        self.conf = conf

    def output_str(self, train_proj, test_proj, unk_rate, entropy): 
        return f'{train_proj},{test_proj},pom,{unk_rate},{entropy}'

    def get_validator(self, projname):
        return PomCrossProjectValidator(projname, self.conf)

class JavaCrossProjectValidator(CrossProjectValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project),
                         JavaTokenizer)
        self.conf = conf

    def output_str(self, train_proj, test_proj, unk_rate, entropy): 
        return f'{train_proj},{test_proj},java,{unk_rate},{entropy}'

    def get_validator(self, projname):
        return JavaCrossProjectValidator(projname, self.conf)

class NextTokenValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, conf.get_next_token_order(), listfile,
                         tokenizer)
        self.testSize = conf.get_next_token_test_size()
        self.minCandidates = conf.get_min_candidates()
        self.maxCandidates = conf.get_max_candidates()

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

    @abstractmethod
    def output_str(self, train_proj, test_proj, unk_rate, entropy):
        raise NotImplementedError()

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
        allkeys = set(list(correct.keys()) + list(incorrect.keys()))
        for token_len in allkeys:
            if rtnstr:
                rtnstr += os.linesep

            n_correct = len(correct[token_len]) if token_len in correct else 0
            n_incorrect = len(incorrect[token_len]) if token_len in incorrect else 0

            rtnstr += self.output_str(nCandidates, token_len, n_correct,
                                      n_incorrect)

        return rtnstr

    def validate(self, executor):
        futures_list = list()
        trainCorpus, testCorpus = self.nextTokenCorpusSplit()
        
        fitter = self.trainModel(trainCorpus)
        for nCandidates in range(self.minCandidates, (self.maxCandidates+1)):
            f = executor.submit(self.guessNextTokenEvaluator, fitter, testCorpus, nCandidates)
            futures_list.append(f)

        return futures_list

class JavaNextTokenValidator(NextTokenValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project),
                         JavaTokenizer)

    def output_str(self, n_candidates, token_len, n_correct, n_incorrect):
        return f'{self.project},java,{n_candidates},{token_len},{n_correct},{n_incorrect}'

class PomNextTokenValidator(NextTokenValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)

    def output_str(self, n_candidates, token_len, n_correct, n_incorrect):
        return f'{self.project},pom,{n_candidates},{token_len},{n_correct},{n_incorrect}'


class TokenizeValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf):
        super().__init__(project, conf, 3, self.listfile)
        self.prefix = conf.get_prefix()
        self.tokenprefix = conf.get_tokenprefix()

    def validate(self, executor):
        futures_list = list()
        mylist = self.listfile

        fhandle = open(mylist, 'r')
        lines = fhandle.readlines()
        fhandle.close()
        for file in lines:
            fname = file.strip()
            t = self.tokenizer(fname, self.prefix, self.tokenprefix, self.versions, self.paths)
            f = executor.submit(t.sentence_tokenize)
            futures_list.append(f)

        return futures_list

class PomTokenizeValidator(TokenizeValidator):
    def __init__(self, project, conf):
        self.tokenizer = PomTokenizer
        self.listfile = conf.get_pomlist(project)
        super().__init__(project, conf)

class JavaTokenizeValidator(TokenizeValidator):
    def __init__(self, project, conf):
        self.tokenizer = JavaTokenizer
        self.listfile = conf.get_srclist(project)
        super().__init__(project, conf)
