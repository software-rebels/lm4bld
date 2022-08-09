import json
import os
import re

from nltk.tokenize import RegexpTokenizer
from abc import ABCMeta, abstractmethod

class AbstractTokenizer(metaclass=ABCMeta):
    def __init__(self, filename, prefix, tokenprefix, ignore_syntax=False,
                 versions=False, paths=False):
        self.filename = filename
        self.prefix = prefix
        self.tokenprefix = tokenprefix
        self.ignore_syntax = ignore_syntax
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
        return self.tokenize_string(strdata)

    def tokenize_string(self, strdata):
        return self.tokenizer().tokenize(strdata)

    # Default case: Do nothing
    def normalize(self, tok):
        return tok

    def tokens_to_sents(self):
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

    def sentence_tokenize(self):
        sents = self.tokens_to_sents()

        outfile = self.get_token_file()
        out_handle = open(outfile, 'w')
        json.dump(sents, out_handle)
        out_handle.close()

        return outfile

    def remove_syntax(self, tokens):
        clean_tokens = list()

        for token_line in tokens:
            clean_tokens.append([token for token in token_line if
                                 not self.is_syntax(token)])

        return clean_tokens

    @abstractmethod
    def is_syntax(self, tokens):
        raise NotImplementedError()

    def load_tokens(self, tarhandle):
        tokenfile = self.get_token_file()
        token_handle = tarhandle.extractfile(tokenfile) if tarhandle else open(tokenfile, 'r', encoding="ISO-8859-1")
        tokens = json.load(token_handle)
        token_handle.close()

        return self.remove_syntax(tokens) if (self.ignore_syntax) else tokens

class PomTokenizer(AbstractTokenizer):
    def comment_re(self):
        return re.compile(r"\<\!\-\-[\s\S]*?\-\-\>")

    def tokenizer(self):
        return RegexpTokenizer('[\"\']|\n|\<\?|\?\>|\<\/|\/\>|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')

    # Not sure if we should do this for Maven
    def normalize(self, tok):
        return tok.lower()

    def is_syntax(self, token):
        return token in ["+", "-", "/", "*", "%", "++", "--", "!", "=", "+=",
                         "-=", "*=", "/=", "%=", "^=", "==", "!=", "<", ">",
                         "<=", ">=", "&&", "||", "?", ":", "&", "|", "^", "~",
                         "<<", ">>", ">>>" ";", "!=", ";", "(", ")", "[", "]",
                         "{", "}", ",", "\"", "\'", "/>", "</","<?", "?>", "."]

class JavaTokenizer(AbstractTokenizer):
    def comment_re(self):
        #return re.compile(r"\/\*[\s\S]*?\*\/|\/\/[\s\S]*?")
        return re.compile(r"(\/\*[\s\S]*?\*\/|\/\/[\s\S]*?)")

    def tokenizer(self):
        return RegexpTokenizer('[\"\']|\n|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')

    def is_syntax(self, token):
        return token in ["+", "-", "/", "*", "%", "++", "--", "!", "=", "+=",
                         "-=", "*=", "/=", "%=", "^=", "==", "!=", "<", ">",
                         "<=", ">=", "&&", "||", "?", ":", "&", "|", "^", "~",
                         "<<", ">>", ">>>" ";", "!=", ";", "(", ")", "[", "]",
                         "{", "}", ",", "\"", "\'", "."]
