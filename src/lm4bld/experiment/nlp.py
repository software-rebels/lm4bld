from lm4bld.experiment.validation import PomCrossFoldValidator
from lm4bld.experiment.validation import JavaCrossFoldValidator
from lm4bld.experiment.validation import PomCrossProjectTrainModelsValidator
from lm4bld.experiment.validation import JavaCrossProjectTrainModelsValidator
from lm4bld.experiment.validation import PomCrossProjectTestModelsValidator
from lm4bld.experiment.validation import JavaCrossProjectTestModelsValidator
from lm4bld.experiment.validation import JavaNextTokenValidator
from lm4bld.experiment.validation import PomNextTokenValidator
from lm4bld.experiment.validation import JavaTokenizeValidator
from lm4bld.experiment.validation import PomTokenizeValidator
from lm4bld.experiment.api import Experiment

class CrossFoldExperiment(Experiment):
    def getPomValidator(self):
        return PomCrossFoldValidator(self.project, self.conf, None)

    def getSrcValidator(self):
        return JavaCrossFoldValidator(self.project, self.conf, None)

class CrossProjectTrainModelsExperiment(Experiment):
    def getSrcValidator(self):
        return JavaCrossProjectTrainModelsValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomCrossProjectTrainModelsValidator(self.project, self.conf)
        
class CrossProjectTestModelsExperiment(Experiment):
    def getSrcValidator(self):
        return JavaCrossProjectTestModelsValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomCrossProjectTestModelsValidator(self.project, self.conf)

class NextTokenExperiment(Experiment):
    def getSrcValidator(self):
        return JavaNextTokenValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomNextTokenValidator(self.project, self.conf)

class TokenizeExperiment(Experiment):
    def getSrcValidator(self):
        return JavaTokenizeValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomTokenizeValidator(self.project, self.conf)
