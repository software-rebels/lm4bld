import re

from nltk.tokenize import RegexpTokenizer
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

class PomTokenizer:
    def __init__(self, filename):
        self.fhandle = open(filename, 'r')

    def removeComments(self, strdata):
        pattern = re.compile(r"\<\!\-\-[\s\S]*?\-\-\>")

        return re.sub(pattern, "", strdata)

    def tokenize(self):
        strdata = self.removeComments(self.fhandle.read())
        tokenizer = RegexpTokenizer(
            '[\"\'][\s\S]*?[\"\']|\n|\<\?|\?\>|\<\/|\/\>|\=|\.|\,|\:|\;|\-|\(|\)|\{|\}|\[|\]|\!|\@|\#|\$|\%|\^|\&|\*|\+|\~|\/|\<|\>|\w+')

        return tokenizer.tokenize(strdata)

    def sentence_tokenize(self):
        sents = list()
        toks = self.tokenize()

        sent = list()
        for tok in toks:
            if tok == "\n":
                sents.append(sent)
                sent = list()
            else:
                sent.append(tok)

        return sents

class PomNGram:
    def __init__(self, tokenizer, order):
        self.tokens = tokenizer.sentence_tokenize()
        self.order = order

    def fit(self):
        self.model = MLE(self.order)
        train, vocab = padded_everygram_pipeline(self.order, self.tokens)
        self.model.fit(train, vocab)

        return self.model
