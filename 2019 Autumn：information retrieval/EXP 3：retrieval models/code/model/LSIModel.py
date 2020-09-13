from collections import Counter
import os
import pickle
from gensim import corpora, models, similarities

class LSIModel:
    
    def __init__(self, corpus, preprocessor, output_filename='.\\lsi', num_topics=500):
        self.num_topics = num_topics
        self.origin_corpus = corpus
        self.docs = []
        self.preprocessor = preprocessor
        dict_suffix = 'ohsu'
        corpus_suffix = 'ohsu'
        self.output_filename = output_filename
        self.dict_filename = output_filename + '\\%s.dict' % dict_suffix
        self.corpus_filename = output_filename + '/%s.mm' % corpus_suffix
        self.lsi_filename = output_filename + '\\%s_%s.lsi' % (corpus_suffix, num_topics)
        self.index_filename = output_filename + '\\%s_%s.lsi.index' % (corpus_suffix, num_topics)
        self.doc2id_filename = output_filename + "\\%s.doc2id.pickle" % corpus_suffix
        self.id2doc_filename = output_filename + "\\%s.id2doc.pickle" % corpus_suffix
        self._create_directories()
        
        
    def _create_directories(self):
        if not os.path.exists(self.output_filename):
            os.makedirs(self.output_filename)
            
            
    def _create_docs_dict(self, docs):
        self.doc2id = dict(zip(docs, range(len(docs))))
        self.id2doc = dict(zip(range(len(docs)), docs))
        pickle.dump(self.doc2id, open(self.doc2id_filename, "wb"))
        pickle.dump(self.id2doc, open(self.id2doc_filename, "wb"))
        
    
    def _load_docs_dict(self):
        self.doc2id = pickle.load(open(self.doc2id_filename, 'rb'))
        self.id2doc = pickle.load(open(self.id2doc_filename, 'rb'))
        
        
    def _generate_dictionary(self):
        print("generating dictionary...")
        documents = []
        for document in self.origin_corpus.documents:
            tokens = self.preprocessor.process(document.content)
            documents.append(tokens)
        self.dictionary = corpora.Dictionary(documents)
        self.dictionary.save(self.dict_filename)
        
        
    def _load_dictionary(self, regenerate=False):
        if not os.path.exists(self.dict_filename) or regenerate is True:
            self._generate_dictionary()
        else:
            self.dictionary = corpora.Dictionary.load(self.dict_filename)
        
        
    def _generate_corpus(self):
        print("generating corpus...")
        self.corpus = []
        corpus_memory_friendly = self._vectorize_corpus(self.origin_corpus, self.dictionary)
        count = 0
        for vector in corpus_memory_friendly:
            self.corpus.append(vector)
            count += 1
            if count % 10000 == 0:
                print("%d vectors processed" % count)
        self._create_docs_dict(self.docs)
        corpora.MmCorpus.serialize(self.corpus_filename, self.corpus)
        
    
    def _vectorize_corpus(self, corpus, dictionary):
        for document in corpus.documents:
            docID = document.docID
            tokens = self.preprocessor.process(document.content)
            self.docs.append(docID)
            yield self.dictionary.doc2bow(tokens)
        
    
    def _vectorize(self, content):
        tokens = self.preprocessor.process(content)
        bow = self.dictionary.doc2bow(tokens)
        return self.lsi[bow]
        
        
    def _load_corpus(self, regenerate=False):
        if not os.path.exists(self.corpus_filename) or regenerate is True:
            self._generate_corpus()
        else:
            self.corpus = corpora.MmCorpus(self.corpus_filename)
            
            
    def _generate_lsi_model(self, regenerate=False):
        print("generating lsi models...")
        if not os.path.exists(self.lsi_filename) or regenerate is True:
            self.lsi = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=self.num_topics)
            self.lsi.save(self.lsi_filename)
            self.index = similarities.MatrixSimilarity(self.lsi[self.corpus])
            self.index.save(self.index_filename)
        elif not os.path.exists(self.index_filename):
            self.lsi = models.LsiModel.load(self.lsi_filename)
            self.index = similarities.MatrixSimilarity(self.lsi[self.corpus])
            self.index.save(self.index_filename)
            
            
    def _load_lsi_model(self, regenerate=False):
        if os.path.exists(self.lsi_filename) and os.path.exists(self.index_filename) and regenerate is False:
            self.lsi = models.LsiModel.load(self.lsi_filename)
            self.index = similarities.MatrixSimilarity.load(self.index_filename)
        else:
            self._generate_lsi_model(regenerate)
    
    def load(self, regenerate=False):
        self._load_dictionary(regenerate)
        self._load_corpus(regenerate)
        self._load_lsi_model(regenerate)
        self._load_docs_dict()
    
    def _get_vector(self, doc):
        vec_bow = None
        try:
            vec_bow = self.corpus[self.doc2id[doc]]
        except KeyError:
            print("Document '%s' does not exist. Have you used the proper string cleaner?" % doc)
        return vec_bow
    
    
    def search(self, query, topK=20, normalized=False):
        query_vector = self._vectorize(query)
        query_sims = self.index[query_vector]
        query_sims = sorted(enumerate(query_sims), key=lambda item: -item[1])[:topK]
        sims = [(self.id2doc[docid], weight) for docid, weight in query_sims]
        return sims
    
    
    def get_similars(self, doc, num_sim=20):
        vec_bow = self._get_vector(doc)
        if vec_bow is None:
            return []
        vec_lsi = self.lsi[vec_bow]
        sims = self.index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])[1:num_sim+1]
        sims = [(self.id2doc[docid], weight) for docid, weight in sims]
        return sims
    
    def get_pairwise_similarity(self, doc1, doc2):
        vec_bow1 = self._get_vector(doc1)
        vec_bow2 = self._get_vector(doc2)
        if vec_bow1 is None or vec_bow2 is None:
            return None
        vec_lsi1 = [val for idx,val in self.lsi[vec_bow1]]
        vec_lsi2 = [val for idx,val in self.lsi[vec_bow2]]
        return cosine(vec_lsi1, vec_lsi2)
        