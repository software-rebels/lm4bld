import json
import os
import re

from nltk.tokenize import RegexpTokenizer
from abc import ABCMeta, abstractmethod

class AbstractTokenizer(metaclass=ABCMeta):
    def __init__(self, filename, prefix, tokenprefix, versions=None,
                 paths=None):
        self.filename = filename
        self.prefix = prefix
        self.tokenprefix = tokenprefix
        self.versions = versions
        self.paths = paths

    def version_re(self):
        return re.compile(r"\d+(?:\.[v\d]+)+(?:[-\w]*)?")

    def path_re(self):
        return re.compile(r"\w+(?:\:\/)?(?:\/[\w\.\*\-]+)+(?:\:[0-9]+)?\/?")

    @abstractmethod
    def comment_re(self):
        raise NotImplementedError()

    @abstractmethod
    def tokenizer(self):
        raise NotImplementedError()

    def get_token_file(self):
        tokenfile = self.filename
        tokenfile = tokenfile.replace(self.prefix, self.tokenprefix) + ".json"

        tokendir = os.path.dirname(tokenfile)
        os.makedirs(tokendir, exist_ok=True)

        return tokenfile 

    def preprocess_strings(self, strdata):
        comment_cleanup = self.comment_re()
        strdata = re.sub(comment_cleanup, "", strdata)

        if (self.paths):
            path_cleanup = self.path_re()
            strdata = re.sub(path_cleanup, "__path_locator__", strdata)

        if (self.versions):
            version_cleanup = self.version_re()
            strdata =re.sub(version_cleanup, "__vers_num__", strdata)

        return strdata

    def tokenize(self):
        fhandle = open(self.filename, 'r', encoding="ISO-8859-1")
        strdata = self.preprocess_strings(fhandle.read())
        fhandle.close()
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

        outfile = self.get_token_file()
        out_handle = open(outfile, 'w')
        json.dump(sents, out_handle)
        out_handle.close()

        return outfile

    def load_tokens(self):
        tokenfile = self.get_token_file()
        token_handle = open(tokenfile, 'r', encoding="ISO-8859-1")
        tokens = json.load(token_handle)
        token_handle.close()

        return tokens

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
        return re.compile(r"(\/\*[\s\S]*?\*\/|\/\/[\s\S]*?)")

    def tokenizer(self):
        return RegexpTokenizer('[\"\']|\n|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')
