import re

from nltk.tokenize import RegexpTokenizer

class PomTokenizer:
    def __init__(self, filename, versions, paths):
        self.fhandle = open(filename, 'r')
        self.versions = versions 
        self.paths = paths

    def preprocess_strings(self, strdata):
        # Comments
        comment_cleanup = re.compile(r"\<\!\-\-[\s\S]*?\-\-\>")
        strdata = re.sub(comment_cleanup, "", strdata)

        # Version strings
        if (self.versions):
            version_cleanup = re.compile(r"\d+(?:\.[v\d]+)+(?:[-\w]*)?")
            strdata =re.sub(version_cleanup, "<VERSNUM>", strdata)

        # Path strings
        if (self.paths):
            path_cleanup = re.compile(r"\w+(?:\/[\w\.\*\-]+)+")
            strdata = re.sub(path_cleanup, "<PATHSTR>", strdata)

        return strdata

    def tokenize(self):
        strdata = self.preprocess_strings(self.fhandle.read())

        tokenizer = RegexpTokenizer(
            '[\"\']|\n|\<\?|\?\>|\<\/|\/\>|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')

        return tokenizer.tokenize(strdata)

    # Not sure if we should do this for Maven
    def normalize(self, tok):
        return tok.lower()

    def sentence_tokenize(self):
        sents = list()
        toks = self.tokenize()

        sent = list()
        for tok in toks:
            if tok == "\n":
                sents.append(sent)
                sent = list()
            else:
                sent.append(self.normalize(tok))

        return sents
