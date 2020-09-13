from collections import Counter
import math

from corpus.Corpus import Corpus
from corpus.Indexer import Indexer
from corpus.Preprocessor import Preprocessor

class BooleanModel:
    
    def __init__(self, indexer):
        self.indexer = indexer
    
    
    def search(self, query, topK=20, normalized=False):
        query_posting_list = self.__vectorize(query)
        if normalized:
            document_scores = self.__documents_normalized_score(query_posting_list)
        else:
            document_scores = self.__documents_score(query_posting_list)

        return document_scores[ : topK]
        
        
    def __vectorize(self, content):
        words = self.indexer.generate_tokens(content)
        posting_list = {}
        for word in words:
            posting_list[word] = 1
        return posting_list
    
    
    def __score_tf(self, posting_list1, posting_list2):        
        score = 0
        for term, tf1 in posting_list1.items():
            if term in posting_list2.keys():
                score += 1
        return score
    
    
    def __score_normalized_tf(self, posting_list1, posting_list2):
        score = 0
        norm1 = len(posting_list1)
        norm2 = len(posting_list2)
        
        for term, tf1 in posting_list1.items():
            if term in posting_list2.keys():
                score += 1
        score = score / (math.sqrt(norm1) * math.sqrt(norm2))
        return score
    
    
    def __documents_score(self, query_posting_list):
        document_scores = Counter()
        for term, query_tf in query_posting_list.items():
            documents = self.indexer.dictionary.get(term, {})
            for docID, _ in documents.items():
                document_scores[docID] += 1
        document_scores = list(document_scores.items())
        document_scores.sort(key=lambda ds: ds[1], reverse=True)
        return document_scores

    
    def __documents_normalized_score(self, query_posting_list):
        document_scores = Counter()
        document_norms = Counter()
        query_norm = len(query_posting_list)
        
        for _, documents in self.indexer.dictionary.items():
            for docID, _ in documents.items():
                document_norms[docID] += 1
        
        for term, query_tf in query_posting_list.items():
            documents = self.indexer.dictionary[term]
            for docID, tf in documents.items():
                document_scores[docID] += 1
        
        query_norm = math.sqrt(query_norm)
        for docID in document_scores:
            document_norm = math.sqrt(document_norms[docID])
            document_scores[docID] /= (document_norm * query_norm)
            
        document_scores = list(document_scores.items())
        document_scores.sort(key=lambda ds: ds[1], reverse=True)
        return document_scores
    