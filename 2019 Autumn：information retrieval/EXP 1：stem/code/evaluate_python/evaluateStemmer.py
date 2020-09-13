import sys
import re
import math
import os
import argparse
from accounter import Accounter
from stemReader import StemReader

MAX_TEST_NUMBER = 30000

def getGoldStems(s):
    t = s.split('\t')
    v = t[1].split('[')[0]
    return (v, v)

def evaluate(stopword_path, goldFile_path, stemFile_path, outputDirectory):
    stopwords_file = stopword_path
    stopwords = []
    f = open(stopwords_file)
    for line in f:
        stopwords.append(line.strip())
    f.close()

    gold_file = goldFile_path
    stem_file = stemFile_path
    fGold = open(gold_file, 'r', encoding='UTF-8')
    fStemReader = StemReader(stem_file) 
    goldHits = Accounter()
    currHits = Accounter()

    sentID = 0
    TP = 0
    FN = 0
    FP = 0

    TPCorpus = 0
    FNCorpus = 0
    FPCorpus = 0

    OI = 0
    OICount = 0
    UI = 0
    UICount = 0
    totalCount = 0

    onlyPunct = re.compile('^\W+$')


    outputPath = outputDirectory
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    result_file = open(outputPath + '\\result.txt', 'w')
    overstemming_file = open(outputPath + '\\overstemming.txt', 'w', encoding='UTF-8')
    understemming_file = open(outputPath + '\\understemming.txt', 'w', encoding='UTF-8')
    low_precise_file = open(outputPath + '\\low_precise.txt', 'w', encoding='UTF-8')
    low_recall_file = open(outputPath + '\\low_recall.txt', 'w', encoding='UTF-8')

    print("Stage #1: statistic...")
    while True:
        (word, stem) = fStemReader.getNextWordAndStems()
        if word == None:
            break
        
        goldLine = fGold.readline()
        if (goldLine.strip() == ''):
            sentID += 1
            continue
        
        (gStem, gStemList) = getGoldStems(goldLine)
        if onlyPunct.match(word) != None:
            continue
            
        word = word.lower()
        stem = stem.lower()
        gStem = gStem.lower()
        if word in stopwords:
            continue
            
        #print(gStem + "\t" + word)
        goldHits.addHit(word, gStem, sentID)
        currHits.addHit(word, stem, sentID)
        
        if (len(stem) > len(gStem)):
            UICount += 1
            UI += len(stem) - len(gStem)
            if not len(gStem) == 0 and (len(stem) - len(gStem)) / len(gStem) > 1:
            	understemming_file.write(stem + "\t" + gStem + "\n")
        elif (len(stem) < len(gStem)):
            OICount += 1
            OI += len(gStem) - len(stem)
            if not len(stem) == 0 and (len(stem) - len(gStem)) / len(stem) < -1:
            	overstemming_file.write(stem + "\t" + gStem + "\n")
        totalCount += 1


    test_query_num = 0
    print("Stage #2: restatistics...")
    fStemReader.reset()
    while True:
        (word, stem) = fStemReader.getNextWordAndStems()
        if word == None:
            break
            
        if word == '':
            continue

        if onlyPunct.match(word) != None:
            continue
        
        if word in stopwords:
            continue
            
        word = word.lower()
        gStem = goldHits.words.get(word, [])
        gStem = [s.lower() for s in gStem]
            
        currWords = set(currHits.stem2words.get(stem, []))
        goldWords = set(goldHits.getWordsByStem(word))
        
        TP += len(currWords.intersection(goldWords))
        FP += len(currWords.difference(goldWords))
        FN += len(goldWords.difference(currWords))
        
        currSentencesID = set(currHits.ids.get(stem, []))
        goldSentencesID = set()
        for s in gStem:
            goldSentencesID = goldSentencesID.union(set(goldHits.ids.get(s, [])))
        if not len(currSentencesID) == 0 and (len(goldSentencesID) - len(currSentencesID)) / len(currSentencesID) > 2:
            low_recall_file.write('stem: ' + str(stem) + '-->' 
        		+ str(currWords) + '-->' + str(len(currSentencesID)) + '\n')
            low_recall_file.write('gStem: ' + str(gStem) + '-->'
            	+ str(goldWords) + '-->' + str(len(goldSentencesID)))
            low_recall_file.write('\n')
            low_recall_file.write('\n')

        if not len(goldSentencesID) == 0 and (len(goldSentencesID) - len(currSentencesID)) / len(goldSentencesID) < - 2:
            low_precise_file.write('stem: ' + str(stem) + '-->'
            	+ str(currWords) + '-->' + str(len(currSentencesID)) + '\n')
            low_precise_file.write('gStem: ' + str(gStem) + '-->'
            	+ str(goldWords) + '-->' + str(len(goldSentencesID)))
            low_precise_file.write('\n')
            low_precise_file.write('\n')

        TPCorpus += len(currSentencesID.intersection(goldSentencesID))
        FPCorpus += len(currSentencesID.difference(goldSentencesID))
        FNCorpus += len(goldSentencesID.difference(currSentencesID))

        test_query_num += 1
        if test_query_num > MAX_TEST_NUMBER:
            break

    result_file.write('matrics #1:\n')
    result_file.write('RI: ' + str((totalCount - OICount - UICount) / totalCount) + '\n')
    result_file.write('OI: ' + str(OICount / totalCount) + '\n')
    result_file.write('AOL: ' + str(OI / OICount) + '\n')
    result_file.write('UI: ' + str(UICount / totalCount) + '\n')
    result_file.write('AUL: ' + str(UI / UICount) + '\n')
    result_file.write('\n')

    result_file.write('matrics #2: \n')
    result_file.write('TP: ' + str(TP) + ', FP: ' + str(FP) + ', FN: ' + str(FN) + '\n')
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * p * r / (p + r)
    result_file.write('Precise: ' + str(p) + '\n')
    result_file.write('Recall:  ' + str(r) + '\n')
    result_file.write('F1:      ' + str(F1) + '\n')
    result_file.write('\n')

    result_file.write('matrics #3: \n')
    result_file.write('TP(Corpus-based): ' + str(TPCorpus) + ', FP(Corpus-based): ' +
    				  str(FPCorpus) + ', FN(Corpus-based): ' + str(FNCorpus) + '\n')
    pCorpus = TPCorpus / (TPCorpus + FPCorpus)
    rCorpus = TPCorpus / (TPCorpus + FNCorpus)
    F1Corpus = 2 * pCorpus * rCorpus / (pCorpus + rCorpus)
    result_file.write('Precise(Corpus-based): ' + str(pCorpus) + '\n')
    result_file.write('Recall(Corpus-based):  ' + str(rCorpus) + '\n')
    result_file.write('F1(Corpus-based):       ' + str(F1Corpus) + '\n')

    fGold.close()
    overstemming_file.close()
    understemming_file.close()
    low_precise_file.close()
    low_recall_file.close()
    result_file.close()

parser = argparse.ArgumentParser(description='eval stemmers with corpus')
parser.add_argument('--stopword', default="C:\\Users\\dell\\Desktop\\stemmerEval-master\\stopwords\\English.snow.txt")
parser.add_argument('--goldFileDirectory', default='G:\\dataset\\corpus\\BNC\\Corpus')
parser.add_argument('--stemFileDirectory', default='G:\\dataset\\corpus\\BNC\\Stem')
parser.add_argument('--outputPath', default='G:\\dataset\\corpus\\BNC\\Stem\\evaluateResult')
p = parser.parse_args()

stopword_path = p.stopword
goldFilePath = p.goldFileDirectory
stemFilePath = p.stemFileDirectory
outputPath = p.outputPath

gold_file_rexp = ".*.bnc.txt"
stem_file_rexp = ".*.bnc.stem.txt$"
gold_files = {}
for file_name in os.listdir(goldFilePath):
    if re.match(gold_file_rexp, file_name):
        domain = file_name.split('.')[0]
        gold_files[domain] = os.path.join(goldFilePath, file_name)

stem_file_info = {}
for root,dirs,files in os.walk(stemFilePath):        
    for file in files: 
        if re.match(stem_file_rexp, file):
            domain = file.split('.')[0]
            file_info = (root, file)
            file_info_list = stem_file_info.get(domain, [])
            if file_info_list == []:
                stem_file_info[domain] = [file_info]
            else:
                stem_file_info[domain].append(file_info)

for domain, file_info in stem_file_info.items():
    for i in range(1, len(file_info)):

        file_path = file_info[i][0]
        file_name = file_info[i][1]
        stem_file_path = os.path.join(file_path, file_name) 

        domain = file_name.split('.')[0]
        gold_file_path = gold_files[domain]

        i = file_path.find('Stem')
        result_path = outputPath + "\\" + domain + "\\" + file_path[i + 4:] + "\\"

        if os.path.exists(result_path):
            continue
        print('evaluating ' + file_name)
        print(stem_file_path + '\t' + gold_file_path + " --> " + result_path)
        evaluate(stopword_path, gold_file_path, stem_file_path, result_path)