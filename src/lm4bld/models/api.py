from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    def __init__(self, order, tokenizer, prefix, tokenprefix, ignore_syntax):
        self.order = order
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.tokenprefix = tokenprefix
        self.ignore_syntax = ignore_syntax
