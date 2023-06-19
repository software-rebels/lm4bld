from abc import ABCMeta, abstractmethod
import importlib
import os
from pathlib import Path
import pickle
import random

from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams

from lm4bld.models.nl import NGramModel
from lm4bld.models.tokenize import JavaTokenizer
from lm4bld.models.tokenize import PomTokenizer

class Validator(metaclass=ABCMeta):
    def __init__(self, project, conf, order, listfile, tokenizer, fitclass):
        self.project = project
        self.order = order
        self.listfile = listfile
        self.tokenizer = tokenizer
        self.prefix = conf.get_prefix()
        self.tokenprefix = conf.get_tokenprefix()
        self.fitclass = fitclass
        self.fitclassname = conf.get_fitclass()
        self.ignore_syntax = conf.get_ignoresyntax()

        random.seed(666) # The best random seed \m/

    def lookup_class(self, mod_name, cname):
        mod = importlib.import_module(mod_name)
        return getattr(mod, cname)

    def load_filenames(self):
        flist = list()
        fhandle = open(self.listfile, 'r', encoding="ISO-8859-1")
        for f in fhandle.readlines():
            flist.append(f.strip())

        fhandle.close()
        return flist

    def trainModel(self, trainCorpus, order):
        fitter = self.fitclass(order, self.tokenizer, self.prefix,
                               self.tokenprefix, self.ignore_syntax)
        fitter.fit(trainCorpus)

        return fitter

    def testModel(self, fitter, testCorpus, order):
        return fitter.unkRate(testCorpus), fitter.crossEntropy(testCorpus), fitter.vocabSize()

    def getProject(self):
        return self.project

    @abstractmethod
    def load_data(self, flist):
        raise NotImplementedError()

    @abstractmethod
    def validate(self, executor):
        raise NotImplementedError()

class NLPValidator(Validator, metaclass=ABCMeta):
    def load_data(self, flist=None):
        if flist is None:
            flist = self.load_filenames()

        return flist

class CrossFoldValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, order, listfile, tokenizer, fitclass):
        super().__init__(project, conf, order, listfile, tokenizer, fitclass)
        self.nfolds = conf.get_nfolds()
        self.niter = conf.get_niter()
        self.minorder = conf.get_minorder()
        self.maxorder = conf.get_maxorder()+1

    @abstractmethod
    def output_str(self, proj, unk_rate, entropy, vocab_size, order, fold, iteration):
        raise NotImplementedError()

    def getFolds(self):
        folds = list()

        for fold in range(self.nfolds):
            folds.append(list())

        myList = self.load_data()
        random.shuffle(myList)

        cntr = 0
        for item in myList:
            folds[cntr % self.nfolds].append(item)
            cntr += 1

        return folds

    def nFoldJob(self, trainCorpus, testCorpus, order, fold, iteration):
        fitter = self.trainModel(trainCorpus, order)
        unk_rate, entropy, vocab_size = self.testModel(fitter, testCorpus, order)

        return self.output_str(self.project, unk_rate, entropy, vocab_size,
                               order, fold, iteration)

    def validate(self, executor):
        my_futures = list()

        for order in range(self.minorder, self.maxorder):
            for iteration in range(self.niter):
                myFolds = self.getFolds()

                for fold in range(self.nfolds):
                    testCorpus = myFolds[fold]

                    trainCorpus = list()
                    for trainIdx in range(self.nfolds):
                        if trainIdx != fold:
                            trainCorpus += myFolds[trainIdx]

                    f = executor.submit(self.nFoldJob, trainCorpus, testCorpus,
                                    order, fold, iteration)
                    my_futures.append(f)

        return my_futures

class PomCrossFoldValidator(CrossFoldValidator):
    def __init__(self, project, conf, order):
        super().__init__(project, conf, order, conf.get_pomlist(project),
                         PomTokenizer, self.lookup_class(conf.get_fitpackage(),
                                                    conf.get_fitclass()))

    def output_str(self, proj, unk_rate, entropy, vocab_size, order, fold,
                   iteration):
        unkline = f'{proj},pom,{self.fitclassname},unk_rate,{order},{fold},{iteration},{unk_rate}'
        entline = f'{proj},pom,{self.fitclassname},entropy,{order},{fold},{iteration},{entropy}'
        vocabline = f'{proj},pom,{self.fitclassname},vocab_size,{order},{fold},{iteration},{vocab_size}'
        return f'{unkline}{os.linesep}{entline}{os.linesep}{vocabline}'

class JavaCrossFoldValidator(CrossFoldValidator):
    def __init__(self, project, conf, order, fitclass=NGramModel):
        super().__init__(project, conf, order, conf.get_srclist(project),
                         JavaTokenizer, fitclass)

    def output_str(self, proj, unk_rate, entropy, vocab_size, order, fold, iteration):
        unkline = f'{proj},java,{self.fitclassname},unk_rate,{order},{fold},{iteration},{unk_rate}'
        entline = f'{proj},java,{self.fitclassname},entropy,{order},{fold},{iteration},{entropy}'
        vocabline = f'{proj},java,{self.fitclassname},vocab_size,{order},{fold},{iteration},{vocab_size}'
        return f'{unkline}{os.linesep}{entline}{os.linesep}{vocabline}'

class CrossProjectTrainModelsValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, conf.get_crossproj_order(), listfile,
                         tokenizer, self.lookup_class(conf.get_fitpackage(),
                                                      conf.get_fitclass()))
        self.modelprefix = conf.get_modelprefix()

    def validate(self, executor):
        futures_list = list()
        futures_list.append(executor.submit(self.trainAndSaveModel))
        return futures_list

    @abstractmethod
    def get_model_fname(self):
        raise NotImplementedError()

    def trainAndSaveModel(self):
        model_fname = self.get_model_fname()
        my_fit = self.trainModel(self.load_data(), self.order)

        Path(os.path.dirname(model_fname)).mkdir(parents=True, exist_ok=True)
        fhandle = open(model_fname, 'wb')
        pickle.dump(my_fit, fhandle)
        fhandle.close()

        return model_fname

class JavaCrossProjectTrainModelsValidator(CrossProjectTrainModelsValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project),
                         JavaTokenizer)

    def get_model_fname(self):
        return f"{self.modelprefix}{os.path.sep}{self.project}-java-{self.fitclassname}.pkl"

class PomCrossProjectTrainModelsValidator(CrossProjectTrainModelsValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)

    def get_model_fname(self):
        return f"{self.modelprefix}{os.path.sep}{self.project}-pom-{self.fitclassname}.pkl"

class CrossProjectTestModelsValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, conf.get_crossproj_order(), listfile,
                         tokenizer, self.lookup_class(conf.get_fitpackage(),
                                                      conf.get_fitclass()))
        self.projects = conf.get_projects()
        self.modelprefix = conf.get_modelprefix()

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
        testCorpus = testValidator.load_data()

        my_fit = self.loadModel()
        unk_rate, entropy = self.testModel(my_fit, testCorpus, self.order)

        return self.output_str(self.project, otherProj, unk_rate, entropy)

    def validate(self, executor):
        futures_list = list()

        for testProjName in self.projects:
            if (testProjName == self.project):
                continue

            f = executor.submit(self.crossProjectJob, testProjName)
            futures_list.append(f)

        return futures_list

class PomCrossProjectTestModelsValidator(CrossProjectTestModelsValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)
        self.conf = conf

    def output_str(self, train_proj, test_proj, unk_rate, entropy):
        return f'{train_proj},{test_proj},pom,{self.fitclassname},{unk_rate},{entropy}'

    def get_validator(self, projname):
        return PomCrossProjectTestModelsValidator(projname, self.conf)

    def get_model_fname(self):
        return f"{self.modelprefix}{os.path.sep}{self.project}-pom-{self.fitclassname}.pkl"

class JavaCrossProjectTestModelsValidator(CrossProjectTestModelsValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project),
                         JavaTokenizer)
        self.conf = conf

    def output_str(self, train_proj, test_proj, unk_rate, entropy):
        return f'{train_proj},{test_proj},java,{self.fitclassname},{unk_rate},{entropy}'

    def get_validator(self, projname):
        return JavaCrossProjectTestModelsValidator(projname, self.conf)

    def get_model_fname(self):
        return f"{self.modelprefix}{os.path.sep}{self.project}-java-{self.fitclassname}.pkl"

class NextTokenValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, conf.get_next_token_order(), listfile,
                         tokenizer, self.lookup_class(conf.get_fitpackage(),
                                                      conf.get_fitclass()))
        self.testSize = conf.get_next_token_test_size()
        self.minCandidates = conf.get_min_candidates()
        self.maxCandidates = conf.get_max_candidates() +1
        self.testRatioThreshold = conf.get_testratiothreshold()

    def nextTokenCorpusSplit(self):
        myList = self.load_filenames()
        if self.testSize/len(myList) > self.testRatioThreshold:
            return None, None

        random.shuffle(myList)

        testCorpus = self.load_data(myList[0:self.testSize])
        trainCorpus = self.load_data(myList[self.testSize:len(myList)])

        return trainCorpus, testCorpus

    def guessNextTokenEvaluator(self, fitter, testCorpus, nCandidates):
        perf_dict = fitter.guessNextTokens(testCorpus, nCandidates)
        rtnstr = ""
        for token_len in perf_dict:
            if rtnstr:
                rtnstr += os.linesep

            myvals = perf_dict[token_len]
            rtnstr += f'{self.project},{self.sourceType()},{self.fitclassname},{nCandidates},{token_len},{myvals[0]},{myvals[1]}'

        return rtnstr

    @abstractmethod
    def sourceType(self):
        raise NotImplementedError()

    def validate(self, executor):
        futures_list = list()
        trainCorpus, testCorpus = self.nextTokenCorpusSplit()

        if trainCorpus is not None and testCorpus is not None:
            fitter = self.trainModel(trainCorpus, self.order)
            for nCandidates in range(self.minCandidates, self.maxCandidates):
                f = executor.submit(self.guessNextTokenEvaluator, fitter,
                                    testCorpus, nCandidates)
                futures_list.append(f)

        return futures_list


class JavaNextTokenValidator(NextTokenValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project),
                         JavaTokenizer)

    def sourceType(self):
        return "java"

class PomNextTokenValidator(NextTokenValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)

    def sourceType(self):
        return "pom"

class GlobalNextTokenValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, conf.get_next_token_order(), listfile,
                         tokenizer, self.lookup_class(conf.get_fitpackage(),
                                                      conf.get_fitclass()))
        self.testSize = conf.get_next_token_test_size()
        self.minCandidates = conf.get_min_candidates()
        self.maxCandidates = conf.get_max_candidates() +1
        self.testRatioThreshold = conf.get_testratiothreshold()

    def nextTokenCorpusSplit(self):
        myList = self.load_filenames()
        if self.testSize/len(myList) > self.testRatioThreshold:
            return None, None

        random.shuffle(myList)

        testCorpus = self.load_data(myList[0:self.testSize])
        trainCorpus = self.load_data(myList[self.testSize:len(myList)])

        return trainCorpus, testCorpus

    def guessNextTokenEvaluator(self, fitter, testCorpus, budget):
        correct_count, incorrect_count = fitter.globalGuessNextTokens(testCorpus, budget)

        return f'{self.project},{self.sourceType()},{self.fitclassname},{budget},{correct_count},{incorrect_count}'

    @abstractmethod
    def sourceType(self):
        raise NotImplementedError()

    def validate(self, executor):
        futures_list = list()
        trainCorpus, testCorpus = self.nextTokenCorpusSplit()

        if trainCorpus is not None and testCorpus is not None:
            fitter = self.trainModel(trainCorpus, self.order)
            for budget in range(1000, 10000, 1000):
                f = executor.submit(self.guessNextTokenEvaluator, fitter,
                                    testCorpus, budget)
                futures_list.append(f)

        return futures_list

class JavaGlobalNextTokenValidator(GlobalNextTokenValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project),
                         JavaTokenizer)

    def sourceType(self):
        return "java"

class PomGlobalNextTokenValidator(GlobalNextTokenValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)

    def sourceType(self):
        return "pom"

class TokenizeValidator(NLPValidator, metaclass=ABCMeta):
    def __init__(self, project, conf, listfile, tokenizer):
        super().__init__(project, conf, None, listfile, tokenizer, None)
        self.versions = conf.get_versions()
        self.paths = conf.get_paths()

    def validate(self, executor):
        futures_list = list()
        mylist = self.listfile

        fhandle = open(mylist, 'r')
        lines = fhandle.readlines()
        fhandle.close()
        for file in lines:
            fname = file.strip()
            t = self.tokenizer(fname, self.prefix, self.tokenprefix,
                               self.ignore_syntax, self.versions, self.paths)
            f = executor.submit(t.sentence_tokenize)
            futures_list.append(f)

        return futures_list

class PomTokenizeValidator(TokenizeValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_pomlist(project), PomTokenizer)

class JavaTokenizeValidator(TokenizeValidator):
    def __init__(self, project, conf):
        super().__init__(project, conf, conf.get_srclist(project), JavaTokenizer)

class LocValidator(NLPValidator, metaclass=ABCMeta):
    @abstractmethod
    def output_str(self, count):
        raise NotImplementedError()

    def load_data(self, flist=None):
        flist = super().load_data(flist)

        sents = list()
        for f in flist:
            t = self.tokenizer(f, self.prefix, self.tokenprefix,
                               self.ignore_syntax)
            sents += t.load_tokens()

        return sents

    def validate(self, executor):
        futures_list = list()
        futures_list.append(executor.submit(self.countlines))

        return futures_list

    def countlines(self):
        return self.output_str(len(self.load_data()))

class PomLocValidator(LocValidator):
    def __init__(self, project, conf, fitclass=NGramModel):
        super().__init__(project, conf, None, conf.get_pomlist(project),
                         PomTokenizer, fitclass)

    def output_str(self, count):
        return f"{self.project},pom,{count}"

class JavaLocValidator(LocValidator):
    def __init__(self, project, conf, fitclass=NGramModel):
        super().__init__(project, conf, None, conf.get_srclist(project),
                         JavaTokenizer, fitclass)

    def output_str(self, count):
        return f"{self.project},java,{count}"

class TokenCountValidator(NLPValidator, metaclass=ABCMeta):
    @abstractmethod
    def output_str(self, count):
        raise NotImplementedError()

    def load_data(self, flist=None):
        flist = super().load_data(flist)

        sents = list()
        for f in flist:
            t = self.tokenizer(f, self.prefix, self.tokenprefix,
                               self.ignore_syntax)
            sents += t.load_tokens()

        return sents

    def validate(self, executor):
        futures_list = list()
        futures_list.append(executor.submit(self.count))

        return futures_list

    def count(self):
        sents = self.load_data()
        c = list(map(len, sents))

        return self.output_str(sum(c))

class PomTokenCountValidator(TokenCountValidator):
    def __init__(self, project, conf, fitclass=NGramModel):
        super().__init__(project, conf, None, conf.get_pomlist(project),
                         PomTokenizer, fitclass)

    def output_str(self, count):
        return f"{self.project},pom,{count}"

class JavaTokenCountValidator(TokenCountValidator):
    def __init__(self, project, conf, fitclass=NGramModel):
        super().__init__(project, conf, None, conf.get_srclist(project),
                         JavaTokenizer, fitclass)

    def output_str(self, count):
        return f"{self.project},java,{count}"

class VocabSizeValidator(NLPValidator, metaclass=ABCMeta):
    @abstractmethod
    def output_str(self, count):
        raise NotImplementedError()

    def load_data(self, flist=None):
        flist = super().load_data(flist)

        tokens = set()
        for f in flist:
            t = self.tokenizer(f, self.prefix, self.tokenprefix,
                               self.ignore_syntax)
            for sent in t.load_tokens():
                if sent:
                    flattened = set(sent) if isinstance(sent[0], str) else set(item for sublist in sent for item in sublist)
                    tokens = tokens.union(flattened)

        return tokens

    def validate(self, executor):
        futures_list = list()
        futures_list.append(executor.submit(self.count))

        return futures_list

    def count(self):
        tokens = self.load_data()

        return self.output_str(len(tokens))

class PomVocabSizeValidator(VocabSizeValidator):
    def __init__(self, project, conf, fitclass=NGramModel):
        super().__init__(project, conf, None, conf.get_pomlist(project),
                         PomTokenizer, fitclass)

    def output_str(self, count):
        return f"{self.project},pom,{count}"

class JavaVocabSizeValidator(VocabSizeValidator):
    def __init__(self, project, conf, fitclass=NGramModel):
        super().__init__(project, conf, None, conf.get_srclist(project),
                         JavaTokenizer, fitclass)

    def output_str(self, count):
        return f"{self.project},java,{count}"
