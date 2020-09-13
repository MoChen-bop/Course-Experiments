import sys
import re
import ntpath
import os

sentRegEx = re.compile(r'<s\b.*?>(.*?)</s>', re.DOTALL)
wRegEx = re.compile(r'(<w[^/<>]*hw="([^\"]*)"[^/<>]*>)(.*?)</w>', re.DOTALL)
posRegEx = re.compile(r'<w[^/<>]*pos="([^\"]*)"[^/<>]*>', re.DOTALL)
typeRegEx = re.compile(r'<[ws]text type="(.*?)"', re.DOTALL)
POSinfo = 1

def extract_from(path):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            extract_from(cur_path)
        else:
            extractWords(cur_path)
            print(cur_path)

def extractWords(file_path):
    f = open(file_path, 'r',  encoding='UTF-8')
    fullData = f.read()
    f.close()
    fileName = ntpath.basename(file_path)
    type = 'unknown'
    tt = typeRegEx.findall(fullData)
    for t in tt:
        if t is not None or len(t) != 0:
            type = t
    label1 = file_path.split('\\')[-3]
    label2 = file_path.split('\\')[-2]
    directory = './Corpus/' + type + '/' + label1 + '/' + label2
    if not os.path.exists(directory):
        os.makedirs(directory)
    outFile = open(directory + '/' + fileName + '.txt', 'w', encoding='UTF-8')
    sents = sentRegEx.findall(fullData)
    for s in sents:
        c = 0
        matches = wRegEx.findall(s)
        for m in matches:
            if m is None or len(m) == 0:
                continue

            p = posRegEx.findall(m[0])
            pos = ''
            if len(p) > 0:
                pos = p[0]
            word = m[2].strip()
            stem = m[1].strip()
            if word == '&amp;':
                continue
            outFile.write(word + '\t' + stem)
            if POSinfo:
                outFile.write('[' + pos + ']')
            outFile.write('\n')
            c = c + 1
        if c > 0:
            outFile.write('\n')

def getFileList(path, file_path_list):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            getFileList(cur_path, file_path_list)
        else:
            file_path_list.append(cur_path)

def combineAllFile(file_dir):
    file_list = os.listdir(file_dir)
    for file in file_list:
        cur_path = os.path.join(file_dir, file)
        if os.path.isdir(cur_path):
            new_file_path_1 = cur_path + ".bnc.txt"
            new_file_path_2 = cur_path + ".bnc.words.txt"
            new_file_1 = open(new_file_path_1, 'w', encoding='UTF-8')
            new_file_2 = open(new_file_path_2, 'w', encoding='UTF-8')
            file_path_list = []
            getFileList(cur_path, file_path_list)
            for file_path in file_path_list:
                f = open(file_path, 'r', encoding='UTF-8')
                for line in f.readlines():
                    new_file_1.write(line)
                    words = line.split('\t')
                    if words[0] == '\n':
                        new_file_2.write(words[0])
                    else:
                        new_file_2.write(words[0] + '\n')
                f.close()
            new_file_2.close()
            new_file_1.close()

def main():
	path = str(sys.argv[1])
	extract_from(path)
	combineAllFile('.\\Corpus')

if __name__ == '__main__':
	main()