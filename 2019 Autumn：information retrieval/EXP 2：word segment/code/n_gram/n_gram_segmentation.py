import argparse
import os

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def has_no_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def extract_n_grams(line, n):
    sub_strs = []
    left = 0
    while left < len(line):
        right = left
        while right < len(line) \
            and right < left + n and is_all_chinese(line[left: right + 1]):
            right += 1
        while right < len(line) and has_no_chinese(line[left:right + 1]):
            right += 1

        sub_strs.append(line[left:right])
        left = right  
    return sub_strs

def write_to_file(grams, f):
    for gram in grams:
        f.write(gram + ' ')
    #f.write('\n')

parser = argparse.ArgumentParser(description='N-gram Chinese word segmentation.')
parser.add_argument('--corpusPath', default='G:\\dataset\\corpus\\WS\\icwb2-data\\training\\pku_training.utf8')
parser.add_argument('--testPath', default='G:\\dataset\\corpus\\WS\\icwb2-data\\testing\\pku_test.utf8')
parser.add_argument('--outputPath', default='G:\\dataset\\corpus\\WS\\result')
p = parser.parse_args()

MAX_WORD_LENGTH = 6

corpus_path = p.corpusPath
test_path = p.testPath
output_path = p.outputPath
N = 6

segmenter_name = 'N-gram'
domain = corpus_path.split('/')[-1].split('.')[0].split('_')[0]
print("domain: " + domain)

output_path += "\\" + segmenter_name 
output_dir = output_path

f = open(test_path, encoding='UTF-8')
for n in range(1, N):
    output_path = output_dir + "\\n-" + str(n) 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path += "\\" + domain + "_segment.txt"
    output_file = open(output_path, 'w', encoding='UTF-8')

    for line in f:
    	grams = extract_n_grams(line, n)
    	write_to_file(grams, output_file)

    output_file.close()
    f.seek(0)

f.close()