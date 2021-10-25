from collections import Counter

class PomModel:
    def __init__(self):
        self.tagmap = {}
        self.tagvalmap = {}
        self.tagvallocmap = {}
        self.attribmap = {}
        self.attribvalmap = {}
        self.attribvallocmap = {}

    def removeNamespace(self, s):
        NS_END = "}"
        spl = s.split(NS_END, 1)
        return spl[-1]

    def addToMap(self, mymap, key, value):
        if key not in mymap:
            mymap[key] = []

        mymap[key].append(value)

    def processTag(self, tag, location):
        tagstr = self.removeNamespace(tag.tag)
        fulltagstr = "%s/%s" % (location, tagstr)

        self.addToMap(self.tagmap, location, tagstr)

        if tag.text is None or not tag.text.strip():
            print("%s/%s" % (location, tagstr))
        else:
            tagcontent = tag.text.strip()
            print("%s=%s" % (fulltagstr, tagcontent))
            self.addToMap(self.tagvalmap, tagstr, tagcontent)
            self.addToMap(self.tagvallocmap, fulltagstr, tagcontent)
            
        # If we have some attribs
        for key in tag.attrib:
            keystr = self.removeNamespace(key)
            valstr = self.removeNamespace(tag.attrib[key])
            print("%s/%s.%s=%s" % (location, tagstr, keystr, valstr))

            fullattribstr = "%s.%s" % (fulltagstr, keystr)

            self.addToMap(self.attribmap, tagstr, keystr)
            self.addToMap(self.attribvalmap, keystr, valstr)
            self.addToMap(self.attribvallocmap, fullattribstr, valstr)

    def guessNextTag(self, context="."):
        taglist = self.tagmap[context]
        hist = Counter(taglist)

        return hist.most_common()
        
    def print(self):
        print(self.tagmap)
