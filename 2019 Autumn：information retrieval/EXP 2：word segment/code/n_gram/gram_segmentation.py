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

def extract_grams(line, n_max):
    sub_strs = {}
    for n in range(1, n_max):
        for i in range(len(line) - n):
            if is_all_chinese(line[i:i+n]):
                if n in sub_strs.keys():
                    sub_strs[n].append(line[i:i+n])
                else:
                    sub_strs[n] = [line[i:i+n]]
    return sub_strs

def write_to_file(grams, f):
	for i in range(1, MAX_WORD_LENGTH):
		if i in grams.keys():
			n_grams = grams[i]
			for gram in n_grams:
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

segmenter_name = 'N-gram'
domain = corpus_path.split('/')[-1].split('.')[0].split('_')[0]
print("domain: " + domain)

output_path += "\\" + segmenter_name 

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path += "\\" + domain + "_segment.txt"
output_file = open(output_path, 'w', encoding='UTF-8')

f = open(test_path, encoding='UTF-8')
for line in f:
	grams = extract_grams(line, MAX_WORD_LENGTH)
	write_to_file(grams, output_file)

f.close()
output_file.close()