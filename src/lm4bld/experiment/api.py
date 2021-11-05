from abc import ABCMeta, abstractmethod
import concurrent.futures

class Experiment(metaclass=ABCMeta):
    def __init__(self, project, executor):
        self.project = project
        self.executor = executor

    @abstractmethod
    def getValidator(self): 
        raise NotImplementedError()

    def createFutures(self):
        return self.getValidator().validate(self.executor)
