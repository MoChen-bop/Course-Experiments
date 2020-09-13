from collections import Counter
import math
import numpy as np
from functools import partial

class TF_IDFModel:
    
    def __init__(self, corpus, indexer, args):
        self.corpus = corpus
        self.indexer = indexer
        if args[0] == 'add':
            self.__TF_IDF = partial(self.__TF_IDF_add, alpha=args[1])
        elif args[0] == 'mul':
            self.__TF_IDF = partial(self.__TF_IDF_mul, alpha=args[1])
        elif args == 'log':
            self.__TF_IDF = self.__TF_IDF_log
    
    
    def search(self, query, topK=20, normalized=False):
        query_posting_list = self.__vectorize(query)
        if normalized:
            document_scores = self.__documents_normalized_score_2(query_posting_list)
        else:
            document_scores = self.__documents_score(query_posting_list)

        return document_scores[ : topK]


    def similarity(self, content1, content2, normalized=False):
        posting_list1 = self.__vectorize(content1)
        posting_list2 = self.__vectorize(content2)
        
        if len(posting_list2) < len(posting_list1):
            temp = posting_list1
            posting_list1 = posting_list2
            posting_list2 = temp
        if normalized:
            return self.__score_normalized_tf_idf(posting_list1, posting_list2)
        else:
            return self.__score_tf_idf(posting_list1, posting_list2)
        
        
    def __TF_IDF_add(self, tf, df, alpha):
        if df == 0:
            idf = 0
        else:
            idf = self.indexer.document_num / df
        return idf * alpha + (1 - alpha) * tf
    
    
    def __TF_IDF_mul(self, tf, df, alpha):
        if df == 0:
            idf = 0
        else:
            idf = self.indexer.document_num / df
        return math.pow(idf, alpha) * math.pow(tf, 1 - alpha)
        
        
    def __TF_IDF_log(self, tf, df):
        if df == 0:
            idf = 0
        else:
            idf = np.log(self.indexer.document_num / df)
        return np.log(1 + tf) * idf       
        
    
    def __vectorize(self, content):
        words = self.indexer.generate_tokens(content)
        posting_list = {}
        for word in words:
            if word in posting_list.keys():
                posting_list[word] += 1
            else:
                posting_list[word] = 1
        return posting_list

    
    def __score_tf_idf(self, posting_list1, posting_list2):        
        score = 0
        for term, tf1 in posting_list1.items():
            df = self.indexer.get_term_DF(term)
            tf_idf1 = self.__TF_IDF(tf1, df)
            if term in posting_list2.keys():
                tf_idf2 = self.__TF_IDF(posting_list2[term], df)
                score +=  tf_idf1 * tf_idf2
        return score
    
    
    def __score_normalized_tf_idf(self, posting_list1, posting_list2):
        score = 0
        norm1 = 0
        norm2 = 0
        for term, tf in posting_list2.items():
            df = self.indexer.get_term_DF(term)
            tf_idf = self.__TF_IDF(tf, df)
            norm2 += tf_idf * tf_idf
        
        for term, tf1 in posting_list1.items():
            df = self.indexer.get_term_DF(term)
            tf_idf1 = self.__tf_idf(tf1, idf)
            norm1 += tf_idf1 * tf_idf1
            if term in posting_list2.keys():
                tf_idf2 = self.__TF_IDF(posting_list2[term], df)
                score += tf_idf1 * tf_idf2

        score = score / (math.sqrt(norm1) * math.sqrt(norm2))
        return score
    
    
    def __documents_score(self, query_posting_list):
        document_scores = Counter()
        for term, query_tf in query_posting_list.items():
            df = self.indexer.get_term_DF(term)
            query_tf_idf = self.__TF_IDF(query_tf, df)
            documents = self.indexer.dictionary.get(term, {})
            for docID, tf in documents.items():
                tf_idf = self.__TF_IDF(tf, df)
                document_scores[docID] += query_tf_idf * tf_idf
        document_scores = list(document_scores.items())
        document_scores.sort(key=lambda ds: ds[1], reverse=True)
        return document_scores
    
    
    def __documents_normalized_score(self, query_posting_list):
        document_scores = Counter()
        document_norms = Counter()
        query_norm = 0
        
        for term, query_tf in query_posting_list.items():
            if term in self.indexer.dictionary.keys():
                df = self.indexer.get_term_DF(term)
                query_tf_idf = self.__TF_IDF(query_tf, df)
                query_norm += query_tf_idf * query_tf_idf
                documents = self.indexer.dictionary[term]
                for docID, tf in documents.items():
                    tf_idf = self.__TF_IDF(tf, df)
                    document_norms[docID] += tf_idf * tf_idf
                    document_scores[docID] += query_tf_idf * tf_idf
        
        query_norm = math.sqrt(query_norm)
        for docID in document_scores:
            document_norm = math.sqrt(document_norms[docID])
            document_scores[docID] /= (document_norm * query_norm)
            
        document_scores = list(document_scores.items())
        document_scores.sort(key=lambda ds: ds[1], reverse=True)
        return document_scores
    
    
    def __documents_normalized_score_2(self, query_posting_list):
        document_scores = Counter()
        document_norms = Counter()
        query_norm = 0
        
        for term, documents in self.indexer.dictionary.items():
            df = self.indexer.get_term_DF(term)
            for docID, tf in documents.items():
                tf_idf = self.__TF_IDF(tf, df)
                document_norms[docID] += tf_idf * tf_idf
        
        for term, query_tf in query_posting_list.items():
            if term in self.indexer.dictionary.keys():
                df = self.indexer.get_term_DF(term)
                query_tf_idf = self.__TF_IDF(query_tf, df)
                query_norm += query_tf_idf * query_tf_idf
                documents = self.indexer.dictionary[term]
                for docID, tf in documents.items():
                    tf_idf = self.__TF_IDF(tf, df)
                    document_scores[docID] += query_tf_idf * tf_idf
        
        query_norm = math.sqrt(query_norm)
        for docID in document_scores:
            document_norm = math.sqrt(document_norms[docID])
            document_scores[docID] /= (document_norm * query_norm)
            
        document_scores = list(document_scores.items())
        document_scores.sort(key=lambda ds: ds[1], reverse=True)
        return document_scores
        