from abc import ABCMeta, abstractmethod
from lm4bld.experiment.validation import PomCrossFoldValidator
from lm4bld.experiment.validation import JavaCrossFoldValidator
from lm4bld.experiment.validation import PomCrossProjectValidator
from lm4bld.experiment.validation import JavaCrossProjectValidator
from lm4bld.experiment.validation import JavaNextTokenValidator
from lm4bld.experiment.validation import PomNextTokenValidator
from lm4bld.experiment.api import Experiment
from lm4bld.nlp.tokenize import PomTokenizer

class CrossFoldExperiment(Experiment):
    def getPomValidator(self):
        return PomCrossFoldValidator(self.project, self.conf, self.currorder)

    def getSrcValidator(self):
        return JavaCrossFoldValidator(self.project, self.conf, self.currorder)

    def createFutures(self):
        futures_list = list()
        for order in range(self.conf.get_minorder(),
                           (self.conf.get_maxorder()+1)):
            self.currorder = order
            pv = self.getPomValidator()
            sv = self.getSrcValidator()
            futures_list += pv.validate(self.executor)
            futures_list += sv.validate(self.executor)

        return futures_list

class CrossProjectExperiment(Experiment):
    def getSrcValidator(self):
        return JavaCrossProjectValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomCrossProjectValidator(self.project, self.conf)
        
class NextTokenExperiment(Experiment):
    def getSrcValidator(self):
        return JavaNextTokenValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomNextTokenValidator(self.project, self.conf)
