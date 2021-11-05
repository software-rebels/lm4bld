from abc import ABCMeta, abstractmethod
from lm4bld.experiment.validation import CrossFoldValidator
from lm4bld.experiment.validation import CrossProjectValidator
from lm4bld.experiment.validation import NextTokenValidator
from lm4bld.experiment.api import Experiment
from lm4bld.nlp.pom import PomTokenizer

projlist = None
def get_proj_map(projmap, order, versions, paths):
    global projlist

    if projlist is None:
        projlist = {}
        for projname in projmap: # confdata[PROJECTS]
            mylist = projmap[projname]
            projlist[projname] = CrossProjectValidator(projname, mylist,
                                                       PomTokenizer, order,
                                                       versions, paths)

    return projlist

class CrossFoldExperiment(Experiment):
    def __init__(self, project, executor, listfile, minorder, maxorder,
                 nfolds, niter, versions, paths):
        super().__init__(project, executor)

        self.listfile = listfile
        self.minorder = minorder
        self.maxorder = maxorder
        self.nfolds = nfolds
        self.niter = niter
        self.versions = versions
        self.paths = paths
        self.currorder = minorder

    def getValidator(self):
        return CrossFoldValidator(self.project, self.listfile, PomTokenizer,
                                  self.currorder, self.versions, self.paths,
                                  self.nfolds, self.niter)

    def createFutures(self):
        futures_list = list()
        for order in range(self.minorder, (self.maxorder+1)):
            self.currorder = order
            validator = self.getValidator()
            futures_list += validator.validate(self.executor)

        return futures_list


class CrossProjectExperiment(Experiment):
    def __init__(self, project, executor, listfile, order, projmap, versions,
                 paths):

        super().__init__(project, executor)
        self.listfile = listfile
        self.order = order
        self.projmap = projmap 
        self.versions = versions
        self.paths = paths
        self.projlist = get_proj_map(projmap, order, versions, paths)

    def getValidator(self):
        cpv = CrossProjectValidator(self.project, self.listfile, PomTokenizer,
                                    self.order, self.versions, self.paths)

        cpv.setProjList(self.projlist)
        return cpv

        
class NextTokenExperiment(Experiment):
    def __init__(self, project, executor, listfile, order, testSize,
                 minCandidates, maxCandidates, versions, paths):
        super().__init__(project, executor)
        self.listfile = listfile
        self.order = order
        self.testSize = testSize
        self.minCandidates = minCandidates
        self.maxCandidates = maxCandidates
        self.versions = versions
        self.paths = paths

    def getValidator(self):
        return NextTokenValidator(self.project, self.listfile,
                                  PomTokenizer, self.order, self.versions,
                                  self.paths, self.testSize, self.minCandidates,
                                  self.maxCandidates)
