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

    def most_common(self, ctype, context):
        myHist = Counter()
        for fmap in self.map:
            mymap = self.map[fmap]
            myHist += Counter(mymap[ctype][context])

        return myHist.most_common()

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

class PomModel(Model):
    def __init__(self, order, tokenizer, prefix, tokenprefix, ignore_syntax):
        super().__init__(order, tokenizer, prefix, tokenprefix, ignore_syntax)
        self.grams = None
        self.map = PomMap()

    def removeNamespace(self, s):
        NS_END = "}"
        spl = s.split(NS_END, 1)
        return spl[-1]

    def processTag(self, filename, tag, location):
        tagstr = self.removeNamespace(tag.tag)
        fulltagstr = "%s/%s" % (location, tagstr)

        self.map.add(filename, CType.TAG, location, tagstr)

        if tag.text is not None and tag.text.strip():
            tagcontent = tag.text.strip()
            #print("%s=%s" % (fulltagstr, tagcontent))
            self.map.add(filename, CType.TAGCONTENT, tagstr, tagcontent)
            self.map.add(filename, CType.TAGCONTENTLOC, fulltagstr, tagcontent)

        # If we have some attribs
        for key in tag.attrib:
            keystr = self.removeNamespace(key)
            valstr = self.removeNamespace(tag.attrib[key])
            #print("%s/%s.%s=%s" % (location, tagstr, keystr, valstr))

            fullattribstr = "%s.%s" % (fulltagstr, keystr)

            self.map.add(filename, CType.ATTRKEY, tagstr, keystr)
            self.map.add(filename, CType.ATTRKEYLOC, fulltagstr, keystr)
            self.map.add(filename, CType.ATTRVAL, keystr, valstr)
            self.map.add(filename, CType.ATTRVALLOC, fullattribstr, valstr)

    def guessNext(self, context=".", ctype=CType.TAG):
        return self.map.most_common(context, ctype)

    def print(self):
        print(self.tagmap)

    def fit(self, flist, filelevel):
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

        for key in tag.attrib:
            # process attrib keys
            keystr = self.removeNamespace(key)
            self.grams.append([CType.ATTRKEYLOC, location, keystr])

            # process attrib vals
            fullattribstr = "%s.%s" % (location, keystr)
            valstr = self.removeNamespace(tag.attrib[key])
            self.grams.append([CType.ATTRVALLOC, fullattribstr, valstr])

        location = "%s/%s" % (location, self.removeNamespace(tag.tag))

        for child in tag:
            self.buildGramsFromTag(etree, child, location)

    def logscore(self, ctype, context, term):
        return self.map.logscore(ctype, context, term)

    def crossEntropy(self, flist, filelevel):
        grams = self.buildGrams(flist)
        sumScore = 0

        for gram in grams:
            gramScore = self.logscore(gram[0], gram[1], gram[2])
            if (gramScore is not None):
                sumScore += gramScore

        return -1 * (sumScore / len(grams))

    def unk_tokens(self, ctype, context, term):
        return self.map.unk_tokens(ctype, context, term)

    def unkRate(self, flist, filelevel):
        grams = self.buildGrams(flist)
        count = 0
        total = 0

        for gram in grams:
            unk_count, tok_count = self.unk_tokens(gram[0], gram[1], gram[2])
            count += unk_count
            total += tok_count

        return count / total

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
