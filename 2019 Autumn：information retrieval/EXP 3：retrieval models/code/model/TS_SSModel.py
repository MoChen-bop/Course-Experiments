import numpy as np
import math
from collections import Counter

from model.TS_SS import TS_SS


class TS_SSModel:
    
    def __init__(self, corpus, indexer, preprocessor, measure='Cosine'):
        self.origin_corpus = corpus
        self.indexer = indexer
        self.preprocessor = preprocessor
        model = TS_SS()
        self.measure_type = measure
        if measure == 'Cosine':
            self.SM = model.Cosine
        elif measure == 'ED':
            self.SM = model.Euclidean
        elif measure == 'TS':
            self.SM = model.Triangle
        elif measure == 'SS':
            self.SM = model.Sector
        elif measure == 'TS-SS':
            self.SM = model
        print("building corpus...")
        self.__build_corpus()
            
    
    def search(self, query, topK=20, normalized=False):
        query_bow = self.__get_bow(query)
        document_scores = self.__document_scores(query_bow)
        return document_scores[ : topK]
        
        
    def __build_corpus(self):
        self.doc_bows = {}
        for term, posting_list in self.indexer.dictionary.items():
            df = len(posting_list)
            for docID, tf in posting_list.items():
                if docID not in self.doc_bows.keys():
                    self.doc_bows[docID] = {}
                self.doc_bows[docID][term] = self.__TF_IDF(tf, df)
    
    
    def __get_bow(self, text):
        tokens = self.preprocessor.process(text)
        bow = {}
        for token in tokens:
            if token in bow:
                bow[token] += 1
            else:
                bow[token] = 1
        
        for term in bow.keys():
            if term in self.indexer.dictionary.keys():
                df = len(self.indexer.dictionary[term])
            else:
                df = 0
            bow[term] = self.__TF_IDF(bow[term], df)
        return bow
    
        
    def __vectorize(self, query_bow, doc_bow):
        terms = set.union(set(query_bow.keys()), doc_bow.keys())
        query_vector = []
        doc_vector = []
        for term in terms:
            if term in query_bow.keys():
                query_vector.append(query_bow[term])
            else:
                query_vector.append(0)
            if term in doc_bow.keys():
                doc_vector.append(doc_bow[term])
            else:
                doc_vector.append(0)
        query_vector = np.array(query_vector)
        doc_vector = np.array(doc_vector)
        return query_vector, doc_vector
        
        
    def __TF_IDF(self, tf, df):
        if df == 0:
            idf = 0
        else:
            idf = np.log(self.indexer.document_num / df)
        return np.log(1 + tf) * idf       
        
    
    def __document_scores(self, query_bow):
        document_scores = Counter()
        
        for docID, doc_bow in self.doc_bows.items():
            query_vector, doc_vector = self.__vectorize(query_bow, doc_bow)
            score = self.SM(query_vector, doc_vector)
            document_scores[docID] = score
        
        document_scores = list(document_scores.items())
        if self.measure_type == 'Cosine':
            document_scores.sort(key=lambda ds: ds[1], reverse=True)
        else:
            document_scores.sort(key=lambda ds: ds[1], reverse=False)
        return document_scores
        