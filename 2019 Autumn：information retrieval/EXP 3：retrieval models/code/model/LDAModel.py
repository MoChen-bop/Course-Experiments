from collections import Counter
from sklearn.neighbors import LSHForest
from gensim import corpora, models, similarities

class LDAModel:
    def __init__(self, corpus, preprocessor, num_topics=20):
        self.origin_corpus = corpus
        self.preprocessor = preprocessor
        self.num_topics = num_topics
        self.docID = []
        self.dictionary = None
        self.corpus = None
        
        
    def build(self):
        print("build corpus...")
        self.__build_corpus()
        print("build model...")
        self.__build_model()
        print("vectorize documents...")
        self.__vectorize_corpus()
        
    
    def search(self, query, topK=20, normalized=False):
        query_vector = self.__vectorize(query)
        scores = self.__document_scores(query_vector)
        return scores
        
        
    def __vectorize(self, text):
        tokens = self.preprocessor.process(text)
        bow = self.dictionary.doc2bow(tokens)
        vector = [x[1] for x in self.model.get_document_topics(bow, 
                                                               minimum_probability=0.0)]
        return vector
        
        
    def __build_corpus(self):
        self.texts = []
        for document in self.origin_corpus.documents:
            docID = document.docID
            tokens = self.preprocessor.process(document.content)
            self.docID.append(docID)
            self.texts.append(tokens)
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        
    
    def __build_model(self):
        self.model = models.ldamodel.LdaModel(self.corpus, 
                                              id2word=self.dictionary,
                                              num_topics=self.num_topics)
    
    
    def __vectorize_corpus(self):
        self.lsh = LSHForest(n_estimators=200, 
                             n_neighbors=self.num_topics)
        self.vectorized_docs = []
        for text in self.texts:
            bow = self.dictionary.doc2bow(text)
            vectorized_doc = [x[1] for x in self.model.get_document_topics(bow, 
                                                                          minimum_probability=0.0)]
            self.vectorized_docs.append(vectorized_doc)
        self.lsh.fit(self.vectorized_docs)
        
    
    
    def __document_scores(self, query_vector):
        distances, indices = self.lsh.kneighbors([query_vector])
        document_scores = Counter()
        for i, distance in enumerate(distances[0]):
            index = indices[0][i]
            document_scores[self.docID[index]] = distance
        
        document_scores = list(document_scores.items())
        return document_scores