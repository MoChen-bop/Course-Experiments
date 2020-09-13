from collections import Counter
import math

class TFModel:
    
    def __init__(self, corpus, indexer):
        self.indexer = indexer
        self.corpus = corpus
        
    
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
            return self.__score_normalized_tf(posting_list1, posting_list2)
        else:
            return self.__score_tf(posting_list1, posting_list2)
        
        
    def __vectorize(self, content):
        words = self.indexer.generate_tokens(content)
        posting_list = {}
        for word in words:
            if word in posting_list.keys():
                posting_list[word] += 1
            else:
                posting_list[word] = 1
        return posting_list

    
    def __score_tf(self, posting_list1, posting_list2):        
        score = 0
        for term, tf1 in posting_list1.items():
            if term in posting_list2.keys():
                score += posting_list2[term] * tf1
        return score
    
    
    def __score_normalized_tf(self, posting_list1, posting_list2):
        score = 0
        norm1 = 0
        norm2 = 0
        for _, tf in posting_list2.items():
            norm2 += tf * tf
        
        for term, tf1 in posting_list1.items():
            norm1 += tf1 * tf1
            if term in posting_list2.keys():
                score += posting_list2[term] * tf1
        score = score / (math.sqrt(norm1) * math.sqrt(norm2))
        return score
    
    
    def __documents_score(self, query_posting_list):
        document_scores = Counter()
        for term, query_tf in query_posting_list.items():
            documents = self.indexer.dictionary.get(term, {})
            for docID, tf in documents.items():
                document_scores[docID] += tf * query_tf
        document_scores = list(document_scores.items())
        document_scores.sort(key=lambda ds: ds[1], reverse=True)
        return document_scores
    
    
    def __documents_normalized_score(self, query_posting_list):
        document_scores = Counter()
        document_norms = Counter()
        query_norm = 0
        
        for term, query_tf in query_posting_list.items():
            if term in self.indexer.dictionary.keys():
                query_norm += query_tf * query_tf
                documents = self.indexer.dictionary[term]
                for docID, tf in documents.items():
                    document_norms[docID] += tf * tf
                    document_scores[docID] += tf * query_tf
            
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
        
        for _, documents in self.indexer.dictionary.items():
            for docID, tf in documents.items():
                document_norms[docID] += tf * tf
        
        for term, query_tf in query_posting_list.items():
            if term in self.indexer.dictionary.keys():
                query_norm += query_tf * query_tf
                documents = self.indexer.dictionary[term]
                for docID, tf in documents.items():
                    document_scores[docID] += tf * query_tf
        
        query_norm = math.sqrt(query_norm)
        for docID in document_scores:
            document_norm = math.sqrt(document_norms[docID])
            document_scores[docID] /= (document_norm * query_norm)
            
        document_scores = list(document_scores.items())
        document_scores.sort(key=lambda ds: ds[1], reverse=True)
        return document_scores
        