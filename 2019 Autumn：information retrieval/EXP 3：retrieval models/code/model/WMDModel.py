import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances
from pyemd import emd

class WMDModel:
    
    def __init__(self, corpus, indexer, preprocessor):
        self.corpus = corpus
        self.indexer = indexer
        self.preprocessor = preprocessor
        self.model_path = "C:\\Users\\dell\\Desktop\\DocumentSimilarity\\Word_Mover_Distance-master\\data\\word_model.mod"
        self.model = Word2Vec.load(self.model_path)

    
    def build(self):
        self.__build_vocabulary()
        self.__vectorize_corpus()
        self.__calculate_word_distance()
        
    
    def search(self, query, topK=20, normalized=False):
        query_vector = self._vectorize(query)
        document_scores = self.__documents_score(query_vector)
        return document_scores[ : topK]
        
        
    def _vectorize(self, content):
        v = self.vectorizer.transform([content])[0]
        v = v.toarray().ravel()
        v = v.astype(np.double)
        try:
            v /= v.sum()
        except:
            print(v)
        return v
        
        
    def __build_vocabulary(self):
        print("building vocabulary...")
        self.vocabulary = []
        for word, _ in self.indexer.dictionary.items():
            if word in self.model.wv.vocab:
                self.vocabulary.append(word)
    
    
    def __vectorize_corpus(self):
        print("vectorizing documents...")
        self.vectorizer = CountVectorizer(vocabulary=self.vocabulary)
        self.docIDs = []
        docs = []
        for document in self.corpus.documents:
            self.docIDs.append(document.docID)
            docs.append(document.content)
        self.corpus_vector = []
        for v in self.vectorizer.transform(docs):
            v = v.toarray().ravel()
            v = v.astype(np.double)
            v /= v.sum()
            self.corpus_vector.append(v)
        
        
    def __calculate_word_distance(self):
        W = np.array([self.model[w] for w in self.vectorizer.get_feature_names()
                         if w in self.model])
        self.words_distance = euclidean_distances(W).astype(np.double)
        self.words_distance /= self.words_distance.max()
        
        
    def __documents_score(self, query_vector):
        documents_score = Counter()
        for index, doc_vector in enumerate(self.corpus_vector):
            docID = self.docIDs[index]
            vector1_ix = np.nonzero(query_vector)
            vector2_ix = np.nonzero(doc_vector)
            union_idx = np.union1d(vector1_ix, vector2_ix)
            vector1 = query_vector[union_idx]
            vector2 = doc_vector[union_idx]
            D = self.words_distance[:,union_idx][union_idx]
            score = emd.emd(vector1, vector2, D)
            documents_score[docID] = score
        
        documents_score = list(documents_score.items())
        documents_score.sort(key=lambda ds: ds[1], reverse=False)
        return documents_score
        