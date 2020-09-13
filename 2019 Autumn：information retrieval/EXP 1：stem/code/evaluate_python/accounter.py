class Accounter:
    
    def __init__(self):
        self.words = {}
        self.ids = {}
        self.stem2words = {}
        
        
    def addHit(self, word, stem, sID):
        if stem == '':
            stem = word
        i = self.words.get(word, '')
        if i == '':
            self.words[word] = [stem]
        elif stem not in i:
            i.append(stem)
            
        o = self.ids.get(stem, None)
        if o == None:
            self.ids[stem] = set([sID])
        elif sID not in o:
            o.add(sID)
            
        s = self.stem2words.get(stem, '')
        if s == '':
            self.stem2words[stem] = [word]
        elif word not in s:
            s.append(word)
            
        
    def getWords(self):
            return self.words
        
        
    def getWordsByStem(self, word):
        stems = self.words.get(word, [])
        words = []
        for s in stems:
            words += self.stem2words.get(s, [])
        return words


    def getSentIDs(self, word):
        st = self.words.get(word, '')
        if (st == ''):
            return set()
        r = set()
        for i in st:
            r = r.union(self.ids.get(i, set()))
        return r


    def print(self):
        print('\nwords: ' + str(self.words))
        print('\nids: ' + str(self.ids))


    def getPaiceStems(self):
        return self.stem2words