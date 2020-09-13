import Stemmer
import os
import re
from functools import partial
from nltk.stem import WordNetLemmatizer

class Preprocessor:
    
    def __init__(self, enable_case_folding=True, enable_remove_stop_words=True,
                enable_stemmer=False, enable_lemmatizer=True, min_length=2):
        self.steps = []
        self.SPLIT_WORDS_PATTERN = re.compile(r'\s|\/|\\|\.|\:|\?|\(|\)|\[|\]|\{|\}|\<|\>|\'|\!|\"|\-|,|;|\$|\*|\%|#')
        self.steps.append(self.__split_words)
        if enable_case_folding:
            self.steps.append(self.__case_folding)
        
        if enable_remove_stop_words:
            self.steps.append(self.__remove_stop_words)
            self.stop_words = {'a', 'able', 'about', 'across', 'after', 'all',
                               'almost', 'also', 'am', 'among', 'an', 'and',
                               'any', 'are', 'as', 'at', 'be', 'because', 'been', 
                               'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 
                               'do', 'does', 'either', 'else', 'ever', 'every', 'for',
                               'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her',
                               'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 
                               'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like',
                               'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither',
                               'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or',
                               'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 
                               'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their',
                               'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 
                               'too', 'twas', 'us', 'wants', 'was', 'we', 'were', 'what', 'when',
                               'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
                               'would', 'yet', 'you', 'your'}
        
        if enable_stemmer:
            self.steps.append(self.__stem)
            self.stemmer = Stemmer.Stemmer('english')
        
        if enable_lemmatizer:
            self.steps.append(self.__lemmatiza)
            self.lemmatizer = WordNetLemmatizer()
            
        if min_length:
            self.steps.append(lambda words: self.__remove_short_words(words, min_length))
            
            
    def process(self, words):
        for i, step in enumerate(self.steps):
            words = list(step(words))
        
        return words
    
    
    def __split_words(self, words):
        return list(filter(lambda word: word != '', self.SPLIT_WORDS_PATTERN.split(words)))
    
    def __case_folding(self, words):
        return map(lambda word: word.casefold(), words)
    
    
    def __remove_stop_words(self, words):
        return filter(lambda word: word not in self.stop_words, words)
    
    
    def __stem(self, words):
        return map(lambda word: self.stemmer.stemWord(word), words)
    
    
    def __lemmatiza(self, words):
        return map(lambda word: self.lemmatizer.lemmatize(word), words)
    
    
    def __remove_short_words(self, words, min_length):
        return filter(lambda word: len(word) >= min_length, words)