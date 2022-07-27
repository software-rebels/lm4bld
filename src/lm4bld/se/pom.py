import xml.etree.ElementTree as ElementTree

from collections import Counter
from enum import Enum, auto
from math import log2
import re

class CType(Enum):
    TAG=auto()
    TAGCONTENT=auto()
    TAGCONTENTLOC=auto()
    ATTRKEY=auto()
    ATTRKEYLOC=auto()
    ATTRVAL=auto()
    ATTRVALLOC=auto()

class PomModel:
    def __init__(self, order):
        self.vocab = set()
        self.tagmap = {}
        self.tagvalmap = {}
        self.tagvallocmap = {}
        self.attribmap = {}
        self.attriblocmap = {}
        self.attribvalmap = {}
        self.attribvallocmap = {}
        self.order = order
        self.gamma = 0.00017
        self.grams = None

    def removeNamespace(self, s):
        NS_END = "}"
        spl = s.split(NS_END, 1)
        return spl[-1]

    def addToMap(self, mymap, key, value):
        if key not in mymap:
            mymap[key] = []

        mymap[key].append(value)
        self.vocab.add(value)

    def processTag(self, tag, location):
        tagstr = self.removeNamespace(tag.tag)
        fulltagstr = "%s/%s" % (location, tagstr)

        self.addToMap(self.tagmap, location, tagstr)

        if tag.text is not None and tag.text.strip():
            tagcontent = tag.text.strip()
            #print("%s=%s" % (fulltagstr, tagcontent))
            self.addToMap(self.tagvalmap, tagstr, tagcontent)
            self.addToMap(self.tagvallocmap, fulltagstr, tagcontent)

        # If we have some attribs
        for key in tag.attrib:
            keystr = self.removeNamespace(key)
            valstr = self.removeNamespace(tag.attrib[key])
            #print("%s/%s.%s=%s" % (location, tagstr, keystr, valstr))

            fullattribstr = "%s.%s" % (fulltagstr, keystr)

            self.addToMap(self.attribmap, tagstr, keystr)
            self.addToMap(self.attriblocmap, fulltagstr, keystr)
            self.addToMap(self.attribvalmap, keystr, valstr)
            self.addToMap(self.attribvallocmap, fullattribstr, valstr)

    def getMapByType(self, ctype):
        mymap = None
        match ctype:
            case CType.TAG:
                mymap = self.tagmap
            case CType.TAGCONTENT:
                mymap = self.tagvalmap
            case CType.TAGCONTENTLOC:
                mymap = self.tagvallocmap
            case CType.ATTRKEY:
                mymap = self.attribmap
            case CType.ATTRKEYLOC:
                mymap = self.attriblocmap
            case CType.ATTRVAL:
                mymap = self.attribvalmap
            case CType.ATTRVALLOC:
                mymap = self.attribvallocmap

        return mymap

    def guessNext(self, context=".", ctype=CType.TAG):
        mymap = getMapByType(ctype)

        hist = Counter(mymap[context])

        return hist.most_common()

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
        self.grams.append([location, self.removeNamespace(tag.tag), CType.TAG])

        # process tag content
        if tag.text is not None and tag.text.strip():
            tagcontent = tag.text.strip()
            self.grams.append([location, tagcontent, CType.TAGCONTENTLOC])

        for key in tag.attrib:
            # process attrib keys
            keystr = self.removeNamespace(key)
            self.grams.append([location, keystr, CType.ATTRKEYLOC])

            # process attrib vals
            fullattribstr = "%s.%s" % (location, keystr)
            valstr = self.removeNamespace(tag.attrib[key])
            self.grams.append([fullattribstr, valstr, CType.ATTRVALLOC])

        location = "%s/%s" % (location, self.removeNamespace(tag.tag))

        for child in tag:
            self.buildGramsFromTag(etree, child, location)

    def logscore(self, context, term, ctype):
        # Lookup context
        mymap = self.getMapByType(ctype)
        freq = 0
        n = 0

        if (context in mymap):
            hist = Counter(mymap[context])
            n = hist.total()

            if (term in hist):
                freq = hist[term]

        return log2((freq + self.gamma) / (n + len(self.vocab)*self.gamma))

    def calcEntropy(self, grams):
        sumScore = 0
        count = 0
        for gram in grams:
            gramScore = self.logscore(gram[0], gram[1], gram[2])
            if (gramScore is not None):
                sumScore += gramScore
            else:
                count += 1

        return -1 * (sumScore / len(grams))

    def crossEntropy(self, flist):
        grams = self.buildGrams(flist)
        return self.calcEntropy(grams)

    def is_unk(self, context, term, ctype):
        mymap = self.getMapByType(ctype)

        rtn = True
        if (context in mymap):
            hist = Counter(mymap[context])
            if (term in hist):
                rtn = False

        return rtn

    def calcUnkRate(self, grams):
        count = 0
        for gram in grams:
            if (self.is_unk(gram[0], gram[1], gram[2])):
                count += 1

        return count / len(grams)

    def unkRate(self, flist):
        grams = self.buildGrams(flist)
        return self.calcUnkRate(grams)

class PomParse:
    def __init__(self, filename, model=None):
        self.etree = ElementTree.parse(filename)
        self.root = self.etree.getroot()

        if model is None:
            self.model = PomModel(-1)
        else:
            self.model = model

    def flatten(self, tag=None, location="."):
        if tag is None:
            tag = self.root

        self.model.processTag(tag, location)

        location = "%s/%s" % (location, self.model.removeNamespace(tag.tag))

        for child in tag:
            self.flatten(child, location)

    def getModel(self):
        return self.model
