import re

from nltk.tokenize import RegexpTokenizer
from abc import ABCMeta, abstractmethod

class AbstractTokenizer(metaclass=ABCMeta):
    def __init__(self, filename, versions, paths):
        self.fhandle = open(filename, 'r', encoding="ISO-8859-1")
        self.versions = versions 
        self.paths = paths

    def version_re(self):
        return re.compile(r"\d+(?:\.[v\d]+)+(?:[-\w]*)?")

    def path_re(self):
        return re.compile(r"\w+(?:\/[\w\.\*\-]+)+")

    @abstractmethod
    def comment_re(self):
        raise NotImplementedError()

    @abstractmethod
    def tokenizer(self):
        raise NotImplementedError()

    def preprocess_strings(self, strdata):
        comment_cleanup = self.comment_re()
        strdata = re.sub(comment_cleanup, "", strdata)

        if (self.versions):
            version_cleanup = self.version_re()
            strdata =re.sub(version_cleanup, "<VERSNUM>", strdata)

        if (self.paths):
            path_cleanup = self.path_re()
            strdata = re.sub(path_cleanup, "<PATHSTR>", strdata)

        return strdata

    def tokenize(self):
        strdata = self.preprocess_strings(self.fhandle.read())
        return self.tokenizer().tokenize(strdata)

    # Default case: Do nothing
    def normalize(self, tok):
        return tok

    def sentence_tokenize(self):
        sents = list()
        toks = self.tokenize()

        sent = list()
        for tok in toks:
            if tok == "\n":
                if sent:
                    sents.append(sent)

                sent = list()
            else:
                sent.append(self.normalize(tok))

        return sents

class PomTokenizer(AbstractTokenizer):
    def comment_re(self):
        return re.compile(r"\<\!\-\-[\s\S]*?\-\-\>")

    def tokenizer(self):
        return RegexpTokenizer('[\"\']|\n|\<\?|\?\>|\<\/|\/\>|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')

    # Not sure if we should do this for Maven
    def normalize(self, tok):
        return tok.lower()

class JavaTokenizer(AbstractTokenizer):
    def comment_re(self):
        return re.compile(r"\/\*[\s\S]*?\*\/|\/\/[\s\S]*?")

    def tokenizer(self):
        return RegexpTokenizer('[\"\']|\n|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')
