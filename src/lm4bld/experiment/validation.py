#import itertools
from abc import ABCMeta, abstractmethod
import concurrent.futures
import pickle
import os
import random

from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams

from lm4bld.nlp.model import NGramModel
from lm4bld.nlp.tokenize import JavaTokenizer
from lm4bld.nlp.tokenize import PomTokenizer

class NLPValidator(metaclass=ABCMeta):
    def __init__(self, project, conf, order, listfile, tokenizer):
        self.project = project
        self.order = order 
        self.listfile = listfile
        self.versions = conf.get_versions()
        self.paths = conf.get_paths()
        self.tokenizer = tokenizer
        self.prefix = conf.get_prefix()
        self.tokenprefix = conf.get_tokenprefix()
        self.modelprefix = conf.get_modelprefix()

    def load_filenames(self):
        flist = list()
        fhandle = open(self.listfile, 'r', encoding="ISO-8859-1")
        for f in fhandle.readlines():
            flist.append(f.strip())

        fhandle.close()
        return flist

    def load_sents(self):
        sents = list()

        for f in self.load_filenames():
            t = self.tokenizer(f, self.prefix, self.tokenprefix)
            sents += t.load_tokens()
        
        return sents

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
        self.minorder = conf.get_minorder()
        self.maxorder = conf.get_maxorder()
        self.maxthreads = conf.get_maxthreads()

    @abstractmethod
    def output_str(self, proj, unk_rate, entropy, order, fold, iteration):
        raise NotImplementedError()

    def getFolds(self):
        foldSents = list()

        for fold in range(self.nfolds):
            foldSents.append(list())

        myList = self.load_sents()
        random.seed(666) # I'm paranoid...
        random.shuffle(myList)

        cntr = 0
        for sent in myList:
            foldSents[cntr % self.nfolds].append(sent)
            cntr += 1

        return foldSents

    def nFoldThread(self, fold, iteration):
        testCorpus = self.myFolds[fold]
        trainCorpus = list()
        for trainIdx in range(self.nfolds):
            if trainIdx != fold:
                trainCorpus += self.myFolds[trainIdx]

        fitter = self.trainModel(trainCorpus)
        unk_rate, entropy = self.testModel(fitter, testCorpus)

        return self.output_str(self.project, unk_rate, entropy, self.order,
                               fold, iteration)


    def nFoldProcess(self, order, iteration):
        self.order = order
        self.myFolds = self.getFolds()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=
                                                         self.maxthreads)

        my_futures = list()
        for fold in range(self.nfolds):
            my_futures.append(executor.submit(self.nFoldThread, fold,
                                              iteration))

        rtnstr = ""
        for future in concurrent.futures.as_completed(my_futures):
            if rtnstr:
                rtnstr += os.linesep
            rtnstr += future.result()

        executor.shutdown()

        return rtnstr

    def validate(self, executor):
        my_futures = list()

        for order in range(self.minorder, (self.maxorder+1)):
            for iteration in range(self.niter):
                f = executor.submit(self.nFoldProcess, order, iteration)
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

class CrossProjectTrainModelsValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, conf.get_crossproj_order(), listfile,
                         tokenizer)

    def validate(self, executor):
        futures_list = list()
        futures_list.append(executor.submit(self.trainAndSaveModel))
        return futures_list

    @abstractmethod
    def get_model_fname(self):
        raise NotImplementedError()

    def trainAndSaveModel(self):
        model_fname = self.get_model_fname()
        my_fit = self.trainModel(self.load_sents())

        fhandle = open(model_fname, 'wb')
        pickle.dump(my_fit, fhandle)
        fhandle.close()

        return model_fname

class JavaCrossProjectTrainModelsValidator(CrossProjectTrainModelsValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project),
                         JavaTokenizer)

    def get_model_fname(self):
        return f"{self.modelprefix}{os.path.sep}{self.project}-java.pkl"

class PomCrossProjectTrainModelsValidator(CrossProjectTrainModelsValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)

    def get_model_fname(self):
        return f"{self.modelprefix}{os.path.sep}{self.project}-pom.pkl"

class CrossProjectTestModelsValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, conf.get_crossproj_order(), listfile,
                         tokenizer)
        self.projects = conf.get_projects()

    @abstractmethod
    def output_str(self, train_proj, test_proj, unk_rate, entropy):
        raise NotImplementedError()

    @abstractmethod
    def get_validator(self, projname):
        raise NotImplementedError()

    def loadModel(self):
        model_fname = self.get_model_fname()
        fhandle = open(model_fname, "rb")
        my_fit = pickle.load(fhandle)
        fhandle.close()

        return my_fit

    def crossProjectJob(self, otherProj):
        testValidator = self.get_validator(otherProj)
        testCorpus = testValidator.load_sents()

        my_fit = self.loadModel()
        unk_rate, entropy = self.testModel(my_fit, testCorpus)

        return self.output_str(self.project, otherProj, unk_rate, entropy)

    def validate(self, executor):
        futures_list = list()

        for testProjName in self.projects:
            if (testProjName == self.project):
                continue

            f = executor.submit(self.crossProjectJob, testProjName)
            futures_list.append(f)
            #print(self.crossProjectJob(testProjName))

        return futures_list

class PomCrossProjectTestModelsValidator(CrossProjectTestModelsValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)
        self.conf = conf

    def output_str(self, train_proj, test_proj, unk_rate, entropy): 
        return f'{train_proj},{test_proj},pom,{unk_rate},{entropy}'

    def get_validator(self, projname):
        return PomCrossProjectTestModelsValidator(projname, self.conf)

    def get_model_fname(self):
        return f"{self.modelprefix}{os.path.sep}{self.project}-pom.pkl"

class JavaCrossProjectTestModelsValidator(CrossProjectTestModelsValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project),
                         JavaTokenizer)
        self.conf = conf

    def output_str(self, train_proj, test_proj, unk_rate, entropy): 
        return f'{train_proj},{test_proj},java,{unk_rate},{entropy}'

    def get_validator(self, projname):
        return JavaCrossProjectTestModelsValidator(projname, self.conf)

    def get_model_fname(self):
        return f"{self.modelprefix}{os.path.sep}{self.project}-java.pkl"

class NextTokenValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, conf.get_next_token_order(), listfile,
                         tokenizer)
        self.testSize = conf.get_next_token_test_size()
        self.minCandidates = conf.get_min_candidates()
        self.maxCandidates = conf.get_max_candidates()

    def nextTokenCorpusSplit(self):
        myList = self.load_filenames()
        random.seed(666)
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
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, 3, listfile, tokenizer)
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
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)

class JavaTokenizeValidator(TokenizeValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project), JavaTokenizer)
