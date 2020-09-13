class StemReader:
    
    def __init__(self, filename):
        self.f = open(filename, 'rb')
    
    def reset(self):
        self.f.seek(0)
        
    def getNextWordAndStems(self):
        stem = ''
        line = self.f.readline()
        if not line:
            return (None, '')
        try:
            line = line.decode('utf-8')
        except:
            line = line.decode('gbk', 'ignore').encode('utf-8')
        line = str(line)
        p = line.strip().split('\t')
        if len(p) == 1:
            return ('', '')
        return (p[0], p[1])