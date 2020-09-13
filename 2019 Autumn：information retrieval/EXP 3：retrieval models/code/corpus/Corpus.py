import os
from corpus.OHSUMED_Extractor import OHSUMED_Extractor
from corpus.Preprocessor import Preprocessor

class Corpus:
    
    def __init__(self, extractor):
        self.extractor = extractor
        self.documents = []
        
        
    def build(self, documents_path):
        documents = []
        if os.path.isdir(documents_path):
            for root, _, file in os.walk(documents_path):
                document_file = os.path.join(root, file)
                documents += self.extractor.extract(document_file)
        else:
            documents += self.extractor.extract(documents_path)
        
        self.documents = documents
    