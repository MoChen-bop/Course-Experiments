import os

class Indexer:
    
    def __init__(self, preprocessor=None, field='content',
                 output_file_path='.\\index'):
        self.preprocessor = preprocessor
        self.output_file_path = output_file_path
        self.dictionary = {}
        self.field_name = field
        self.index_file_name = self.output_file_path + "\\" + self.field_name + ".index"
        self.document_num = 0
        if field == 'content':
            self.field = 3
        elif field == 'authors':
            self.field = 4
        elif field == 'keywords':
            self.field = 5
        elif field == 'publication':
            self.field = 7
        
        
    def index(self, corpus, reIndex=False):
        self.document_num = len(corpus.documents)
        if not reIndex and os.path.exists(self.index_file_name):
            print('loading index from path: %s' % self.index_file_name)
            self.__load_index()
        else:
            print('rebuilding index...')
            self.__build_index(corpus)
            
        
    def get_posting_list(self, word):
        if word in self.dictionary.keys():
            return dictionary[word]
        else:
            return { }
    
    
    def get_term_DF(self, word):
        if word in self.dictionary.keys():
            return len(self.dictionary[word])
        else:
            return 0
    
    
    def get_doc_TF(self, docID, word):
        posting_list = get_posting_list(word)
        if docID in posting_list.keys():
            return posting_list[docID]
        else:
            return 0
        
    
    def generate_tokens(self, words_stream):
        if self.preprocessor is not None:
            words = self.preprocessor.process(words_stream)
        return words
    
        
    def __generate_tokens(self, words_stream):
        if self.preprocessor is not None:
            words = self.preprocessor.process(words_stream)
        return words
    
    
    def __add_into_dictionary(self, docID, words):
        for word in words:
            if word in self.dictionary.keys():
                posting_list = self.dictionary[word]
                if docID in posting_list.keys():
                    posting_list[docID] += 1
                else:
                    posting_list[docID] = 1
            else:
                posting_list = {docID : 1}
                self.dictionary[word] = posting_list            
    
        
    def __build_index(self, corpus):
        for document in corpus.documents:
            docID = document.docID
            words_stream = document[self.field]
            if isinstance(words_stream, list):
                words = words_stream
            else:
                words = self.__generate_tokens(words_stream)
            self.__add_into_dictionary(docID, words) 
        self.__flush_index_entry()



    def __load_index(self):
        f = open(self.index_file_name)
        for line in f:
            term, _, docs = line.strip().split('\t')
            self.dictionary[term] = {}
            for doc in docs.split(','):
                docID, tf = doc.split('|')
                docID = int(docID)
                self.dictionary[term][docID] = int(tf)
        f.close()


    def __flush_index_entry(self):
        if not os.path.exists(self.output_file_path):
            os.makedirs(self.output_file_path)
        index_file = open(self.index_file_name, 'w', encoding='utf-8')
        for term, posting_list in self.dictionary.items():
            self.__write_index_entry(index_file, term, posting_list)
        index_file.close()
        
        
    def __write_index_entry(self, file, term, posting_list):
        posting = list(map(lambda e: '{}|{}'.format(e[0], e[1]),
                           posting_list.items()))
        line = '{}\t{}\t{}\n'.format(term, str(len(posting_list)), ','.join(posting))
        file.write(line)