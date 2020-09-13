import sys
import re
import math
import os
import argparse

def getGoldStems(s):
    t = s.split('\t')
    v = t[1].split('(')
    if len(v) == 1:
        x = v[0].strip()
        return (t[0], x, x)
    return (t[0], v[0], v[1].split(','))

def isStopWord(stem, stopwords):
    s = stem.split('[')
    if len(s) > 1:
        s1 = '['.join(s[0:-1])
        return s1 in stopwords
    return s in stopwords

class word2sentence:
    
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
        r = ''
        for s in stems:
            r += ' ' + s + ' -> ' + str(self.stem2words.get(s, []))
        return r
    
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

parser = argparse.ArgumentParser(description='eval stemmers with corpus')
parser.add_argument('--stopword', default="C:\\Users\\dell\\Desktop\\stemmerEval-master\\stopwords\\English.snow.txt")
parser.add_argument('--gold', default='')
parser.add_argument('--domain', default='')
parser.add_argument('--fullSentsPath', default='')
p = parser.parse_args()

stopword_file = p.stopword
stopwords = []
f = open(stopword_file, 'r', encoding='UTF-8')
for line in f:
    stopwords.append(line.strip())
f.close()

fGold_name = p.gold
fGold = open(fGold_name, 'r', encoding='UTF-8')

domain = p.domain
fullSents = p.fullSentsPath
fullSentsPath =  fullSents +  '/' + domain
if not os.path.exists(fullSentsPath):
    os.makedirs(fullSentsPath)
goldSentFilename = fullSentsPath + '/' + domain + '.GoldSents.txt'

fDebug = open('sentLists.txt', 'w', encoding='UTF-8')

sentID = 0
currSent = []
inputWord = ''
stems = []
stem = ''
first = 1
goldHits = word2sentence()

goldLemmas = {}
onlyPunct = re.compile('^\W+$')

for goldLine in fGold:
    if (goldLine.strip() == ''):
        prefix = str(int(sentID / 1000))
        if not os.path.exists(fullSentsPath + '/' + prefix):
            os.makedirs(fullSentsPath + '/' + prefix)
        currSentFile = open(fullSentsPath + '/' + prefix + '/' + str(sentID) + '.txt', 'w', encoding='UTF-8')
        currSentFile.write(' '.join(currSent))
        currSentFile.close()
        
        fDebug.write('\n\nsentID:' + str(sentID) + '\n')
        fDebug.write('\n'.join(currSent))
        
        currSent = []
        
        sentID += 1
        continue
        
    (word, gStem, gStemList) = getGoldStems(goldLine)
    currSent.append(word.strip())
    if onlyPunct.match(word) != None:
        continue
        
    if isStopWord(gStem.lower(), stopwords):
        continue
    goldHits.addHit(word, gStem, sentID)
    
fDebug.close()
fGold.close()

goldSentFile = open(goldSentFilename, 'w', encoding='UTF-8')

for w in goldHits.getWords():
    sIDsGold = goldHits.getSentIDs(w)
    
    goldSentFile.write(w + '\t' + ",".join(str(e) for e in sIDsGold) + '\n')
    continue
goldSentFile.close()