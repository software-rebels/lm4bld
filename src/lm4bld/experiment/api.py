from abc import ABCMeta, abstractmethod
import concurrent.futures

class Experiment(metaclass=ABCMeta):
    def __init__(self, project, executor, conf):
        self.project = project
        self.executor = executor
        self.conf = conf

    @abstractmethod
    def getPomValidator(self): 
        raise NotImplementedError()

    @abstractmethod
    def getSrcValidator(self):
        raise NotImplementedError()

    def createFutures(self):
        futures_list = list()
        if self.conf.get_process_poms():
            futures_list += self.getPomValidator().validate(self.executor)
        if self.conf.get_process_source():
            futures_list += self.getSrcValidator().validate(self.executor)

        return futures_list
