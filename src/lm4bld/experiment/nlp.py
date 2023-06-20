from lm4bld.experiment.validation import PomCrossFoldValidator
from lm4bld.experiment.validation import JavaCrossFoldValidator
from lm4bld.experiment.validation import PomCrossProjectTrainModelsValidator
from lm4bld.experiment.validation import JavaCrossProjectTrainModelsValidator
from lm4bld.experiment.validation import PomCrossProjectTestModelsValidator
from lm4bld.experiment.validation import JavaCrossProjectTestModelsValidator
from lm4bld.experiment.validation import JavaNextTokenValidator
from lm4bld.experiment.validation import PomNextTokenValidator
from lm4bld.experiment.validation import JavaGlobalNextTokenValidator
from lm4bld.experiment.validation import PomGlobalNextTokenValidator
from lm4bld.experiment.validation import JavaTokenizeValidator
from lm4bld.experiment.validation import PomTokenizeValidator
from lm4bld.experiment.validation import JavaLocValidator
from lm4bld.experiment.validation import PomLocValidator
from lm4bld.experiment.validation import JavaTokenCountValidator
from lm4bld.experiment.validation import PomTokenCountValidator
from lm4bld.experiment.validation import JavaVocabSizeValidator
from lm4bld.experiment.validation import PomVocabSizeValidator
from lm4bld.experiment.validation import JavaHoldoutValidator
from lm4bld.experiment.validation import PomHoldoutValidator
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

class GlobalNextTokenExperiment(Experiment):
    def getSrcValidator(self):
        return JavaGlobalNextTokenValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomGlobalNextTokenValidator(self.project, self.conf)

class TokenizeExperiment(Experiment):
    def getSrcValidator(self):
        return JavaTokenizeValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomTokenizeValidator(self.project, self.conf)

class LocExperiment(Experiment):
    def getSrcValidator(self):
        return JavaLocValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomLocValidator(self.project, self.conf)

class TokenCountExperiment(Experiment):
    def getSrcValidator(self):
        return JavaTokenCountValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomTokenCountValidator(self.project, self.conf)

class VocabSizeExperiment(Experiment):
    def getSrcValidator(self):
        return JavaVocabSizeValidator(self.project, self.conf)

    def getPomValidator(self):
        return PomVocabSizeValidator(self.project, self.conf)

class HoldoutExperiment(Experiment):
    def getSrcValidator(self):
        return JavaHoldoutValidator(self.project, self.conf, None)

    def getPomValidator(self):
        return PomHoldoutValidator(self.project, self.conf, None)
