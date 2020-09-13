from collections import namedtuple

class OHSUMED_Extractor():
    def __init__(self):
        pass
    
    
    def extract(self, documents_file):
        file = open(documents_file, 'r', encoding='utf-8')
        Document = namedtuple('Document', ['docID', 'sID', 'title',  'content', 
                                           'authors', 'keyswords', 'pType', 'publication'])
        documents = []
        not_finish = True
        
        docID = 0
        sID = 0
        title = ''
        content = ''
        authors = []
        keywords = []
        pType = ''
        publication = ''
        source = ''
        
        have_content = False
        while not_finish:
            line = file.readline()
            if line == None or len(line) < 2:
                not_finish = False
                break
            tag = line[1]
            if tag == 'I':
                sID = int(line[3:])
            elif tag == 'U':
                line = file.readline()
                docID = int(line.strip())
            elif tag == 'S':
                line = file.readline()
                publication = [line.strip()]
            elif tag == 'M':
                line = file.readline()
                line = line.strip().strip('.')
                keywords = [word.split('/')[0].strip() for word in line.split(';')]
            elif tag == 'T':
                line = file.readline()
                title = line.strip()
            elif tag == 'P':
                line = file.readline()
                pType = line.strip()
            elif tag == 'W':
                have_content = True
                line = file.readline()
                content = line.strip()
            elif tag == 'A':
                line = file.readline().strip().strip('.')
                authors = [ author.strip() for author in line.split(';')]
                if have_content:
                    documents.append(Document(docID, sID, title, content, 
                                              authors, keywords, pType, publication))
                have_content = False
        file.close()
        return documents
                
                