import xml.etree.ElementTree as ElementTree

from collections import Counter
from enum import Enum, auto
from math import log2
import re

from lm4bld.models.api import Model

class CType(Enum):
    TAG=auto()
    TAGCONTENT=auto()
    TAGCONTENTLOC=auto()
    ATTRKEY=auto()
    ATTRKEYLOC=auto()
    ATTRVAL=auto()
    ATTRVALLOC=auto()

class PomMap:
    def __init__(self):
        self.gamma = 0.00017
        self.map = {}
        self.vocab = set()

    def initMapLayer(self, filename):
        self.map[filename] = {}
        self.map[filename][CType.TAG] = {}
        self.map[filename][CType.TAGCONTENT] = {}
        self.map[filename][CType.TAGCONTENTLOC] = {}
        self.map[filename][CType.ATTRKEY] = {}
        self.map[filename][CType.ATTRKEYLOC] = {}
        self.map[filename][CType.ATTRVAL] = {}
        self.map[filename][CType.ATTRVALLOC] = {}

    def add(self, filename, ctype, key, value):
        if filename not in self.map:
            self.initMapLayer(filename)

        if key not in self.map[filename][ctype]:
            self.map[filename][ctype][key] = []

        self.map[filename][ctype][key].append(value)
        self.vocab.add(value)

    def most_common(self, ctype, context, nCandidates):
        myHist = Counter()
        for fmap in self.map:
            mymap = self.map[fmap]
            if (context in mymap[ctype]):
                myHist += Counter(mymap[ctype][context])

        rtn = []
        for t in myHist.most_common(nCandidates):
            rtn.append(t[0])

        return rtn


    def logscore(self, ctype, context, term):
        freq = 0
        n = 0

        for fmap in self.map:
            mymap = self.map[fmap]
            if (context in mymap[ctype]):
                n += 1

                if (term in mymap[ctype][context]):
                    freq += 1

        return log2((freq + self.gamma) / (n + len(self.vocab)*self.gamma))

    def unk_tokens(self, ctype, context, term):
        rtn = 1

        for fmap in self.map:
            mymap = self.map[fmap]
            if (context in mymap[ctype] and term in mymap[ctype][context]):
                rtn = 0
                break

        return [rtn, 1]

    def flattened_view(self, ctype):
        rtn = {}

        for fmap in self.map:
            mymap = self.map[fmap]
            rtn |= mymap[ctype]

        return rtn

class PomModel(Model):
    def __init__(self, order, tokenizer, prefix, tokenprefix, ignore_syntax):
        super().__init__(order, tokenizer, prefix, tokenprefix, ignore_syntax)
        self.grams = None
        self.map = PomMap()

    def removeNamespace(self, s):
        NS_END = "}"
        spl = s.split(NS_END, 1)
        return spl[-1]

    def processTagName(self, filename, location, tname):
        self.map.add(filename, CType.TAG, location, tname)

    def processPayload(self, filename, tagstr, fulltagstr, payload):
        self.map.add(filename, CType.TAGCONTENT, tagstr, payload)
        self.map.add(filename, CType.TAGCONTENTLOC, fulltagstr, payload)

    def processAttrKey(self, filename, tagstr, fulltagstr, attrkey):
        self.map.add(filename, CType.ATTRKEY, tagstr, attrkey)
        self.map.add(filename, CType.ATTRKEYLOC, fulltagstr, attrkey)

    def processAttrVal(self, filename, attrkey, fullattrkey, attrval):
        self.map.add(filename, CType.ATTRVAL, attrkey, attrval)
        self.map.add(filename, CType.ATTRVALLOC, fullattrkey, attrval)

    def processTag(self, filename, tag, location):
        tagstr = self.removeNamespace(tag.tag)
        fulltagstr = "%s/%s" % (location, tagstr)

        self.processTagName(filename, location, tagstr)

        if tag.text is not None and tag.text.strip():
            tagcontent = tag.text.strip()
            self.processPayload(filename, tagstr, fulltagstr, tagcontent)

        # If we have some attribs
        for key in tag.attrib:
            keystr = self.removeNamespace(key)
            valstr = tag.attrib[key]
            #print("%s/%s.%s=%s" % (location, tagstr, keystr, valstr))

            fullattribstr = "%s.%s" % (fulltagstr, keystr)

            self.processAttrKey(filename, tagstr, fulltagstr, keystr)
            self.processAttrVal(filename, keystr, fullattribstr, valstr)

    def print(self):
        print(self.tagmap)

    def fit(self, flist):
        for f in flist:
            pp = PomParse(f, self)
            pp.flatten()

    def buildGrams(self, flist):
        if (self.grams is not None):
            return self.grams

        self.grams = []

        for f in flist:
            etree = ElementTree.parse(f)
            self.buildGramsFromTag(etree)

        return self.grams

    def buildGramsFromTag(self, etree, tag=None, location="."):
        if tag is None:
            tag = etree.getroot()

        # process tag
        self.grams.append([CType.TAG, location, self.removeNamespace(tag.tag)])

        # process tag content
        if tag.text is not None and tag.text.strip():
            tagcontent = tag.text.strip()
            self.grams.append([CType.TAGCONTENTLOC, location, tagcontent])

        # Update location context
        location = "%s/%s" % (location, self.removeNamespace(tag.tag))

        for key in tag.attrib:
            # process attrib keys
            keystr = self.removeNamespace(key)
            self.grams.append([CType.ATTRKEYLOC, location, keystr])

            # process attrib vals
            fullattribstr = "%s.%s" % (location, keystr)
            valstr = tag.attrib[key]
            self.grams.append([CType.ATTRVALLOC, fullattribstr, valstr])

        for child in tag:
            self.buildGramsFromTag(etree, child, location)

    def logscore(self, ctype, context, term):
        return self.map.logscore(ctype, context, term)

    def crossEntropy(self, flist):
        grams = self.buildGrams(flist)
        sumScore = 0

        for gram in grams:
            gramScore = self.logscore(gram[0], gram[1], gram[2])
            if (gramScore is not None):
                sumScore += gramScore

        return -1 * (sumScore / len(grams))

    def unk_tokens(self, ctype, context, term):
        return self.map.unk_tokens(ctype, context, term)

    def unkRate(self, flist):
        grams = self.buildGrams(flist)
        count = 0
        total = 0

        for gram in grams:
            unk_count, tok_count = self.unk_tokens(gram[0], gram[1], gram[2])
            count += unk_count
            total += tok_count

        return count / total

    def processGramForNextTokenExp(self, gram, nCandidates):
        correct = {}
        incorrect = {}
        token = gram[-1]
        guesses = self.map.most_common(gram[0], gram[1], nCandidates)
        if (token in guesses):
            if len(token) not in correct:
                correct[len(token)] = list()

            correct[len(token)].append(token)

        else:
            if len(token) not in incorrect:
                incorrect[len(token)] = list()

            incorrect[len(token)].append(token)

        return correct, incorrect

    def guessNextTokens(self, testCorpus, nCandidates):
        correct = {}
        incorrect = {}

        grams = self.buildGrams(testCorpus)

        for gram in grams:
            if self.tokenizer.is_syntax(None, gram[2]) or gram[2].isnumeric() or gram[2].isspace():
                continue

            c, i = self.processGramForNextTokenExp(gram, nCandidates)

            for tlen in c:
                if tlen not in correct:
                    correct[tlen] = list()

                correct[tlen].extend(c[tlen])

            for tlen in i:
                if tlen not in incorrect:
                    incorrect[tlen] = list()

                incorrect[tlen].extend(i[tlen])

        rtn = {}
        allkeys = set(list(correct.keys()) + list(incorrect.keys()))
        for token_len in allkeys:
            n_correct = len(correct[token_len]) if token_len in correct else 0
            n_incorrect = len(incorrect[token_len]) if token_len in incorrect else 0
            rtn[token_len] = [n_correct, n_incorrect]

        return rtn

class AblatePayloadPomModel(PomModel):
    def logscore(self, ctype, context, term):
        return None if (ctype is CType.TAGCONTENT or ctype is CType.TAGCONTENTLOC) else super().logscore(ctype, context, term)

    def unk_tokens(self, ctype, context, term):
        return [0, 0] if (ctype is CType.TAGCONTENT or ctype is CType.TAGCONTENTLOC) else super().unk_tokens(ctype, context, term)

class AblateAttrKeyPomModel(PomModel):
    def logscore(self, ctype, context, term):
        return None if (ctype is CType.ATTRKEY or ctype is CType.ATTRKEYLOC) else super().logscore(ctype, context, term)

    def unk_tokens(self, ctype, context, term):
        return [0, 0] if (ctype is CType.ATTRKEY or ctype is CType.ATTRKEYLOC) else super().unk_tokens(ctype, context, term)

class AblateAttrValPomModel(PomModel):
    def logscore(self, ctype, context, term):
        return None if (ctype is CType.ATTRVAL or ctype is CType.ATTRVALLOC) else super().logscore(ctype, context, term)

    def unk_tokens(self, ctype, context, term):
        return [0, 0] if (ctype is CType.ATTRVAL or ctype is CType.ATTRVALLOC) else super().unk_tokens(ctype, context, term)

class AblateTagPomModel(PomModel):
    def logscore(self, ctype, context, term):
        return None if (ctype is CType.TAG) else super().logscore(ctype, context, term)

    def unk_tokens(self, ctype, context, term):
        return [0, 0] if (ctype is CType.TAG) else super().unk_tokens(ctype, context, term)

class PomParse:
    def __init__(self, filename, model=None):
        self.filename = filename
        self.etree = ElementTree.parse(filename)
        self.root = self.etree.getroot()

        if model is None:
            self.model = PomModel(None)
        else:
            self.model = model

    def flatten(self, tag=None, location="."):
        if tag is None:
            tag = self.root

        self.model.processTag(self.filename, tag, location)

        location = "%s/%s" % (location, self.model.removeNamespace(tag.tag))

        for child in tag:
            self.flatten(child, location)

    def getModel(self):
        return self.model
