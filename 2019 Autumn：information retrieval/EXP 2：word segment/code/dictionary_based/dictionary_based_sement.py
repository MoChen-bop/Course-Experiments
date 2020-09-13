import argparse
import os

from dictionary_based_segmenter import Segmenter


def write_to_file(grams, f):
    for gram in grams:
        f.write(gram + ' ')
    #f.write('\n')

parser = argparse.ArgumentParser(description='N-gram Chinese word segmentation.')
parser.add_argument('--corpusPath', default='G:\\dataset\\corpus\\WS\\icwb2-data\\training\\pku_training.utf8')
parser.add_argument('--testPath', default='G:\\dataset\\corpus\\WS\\icwb2-data\\testing\\pku_test.utf8')
parser.add_argument('--outputPath', default='G:\\dataset\\corpus\\WS\\result')
p = parser.parse_args()

MAX_WORD_LENGTH = 10

corpus_path = p.corpusPath
test_path = p.testPath
output_path = p.outputPath
n = 10

segmenter = Segmenter(corpus_path)

segmenter_name = 'Dictionary-based'
domain = corpus_path.split('/')[-1].split('.')[0].split('_')[0]
print("domain: " + domain)
output_path += "\\" + segmenter_name 
output_dir = output_path

################################################################
''' 1#: FMM
'''

f = open(test_path, encoding='UTF-8')

output_path = output_dir + "\\FMM"
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path += "\\" + domain + "_segment.txt"
output_file = open(output_path, 'w', encoding='UTF-8')

for line in f:
	grams = segmenter.longest_match_segment(line, n)
	write_to_file(grams, output_file)
output_file.close()

##############################################################
f.seek(0)

''' 2#: FmM
'''

f = open(test_path, encoding='UTF-8')

output_path = output_dir + "\\FmM"
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path += "\\" + domain + "_segment.txt"
output_file = open(output_path, 'w', encoding='UTF-8')

for line in f:
	grams = segmenter.shortest_match_segment(line, n)
	write_to_file(grams, output_file)
output_file.close()

#############################################################
f.seek(0)

''' 3#: BMM
'''

f = open(test_path, encoding='UTF-8')

output_path = output_dir + "\\BMM"
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path += "\\" + domain + "_segment.txt"
output_file = open(output_path, 'w', encoding='UTF-8')

for line in f:
	grams = segmenter.longest_match_segment_reverse(line, n)
	write_to_file(grams, output_file)
output_file.close()

#############################################################
f.seek(0)

''' 4#: BmM
'''

f = open(test_path, encoding='UTF-8')

output_path = output_dir + "\\BmM"
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path += "\\" + domain + "_segment.txt"
output_file = open(output_path, 'w', encoding='UTF-8')

for line in f:
	grams = segmenter.shortest_match_segment_reverse(line, n)
	write_to_file(grams, output_file)
output_file.close()


#############################################################
f.seek(0)

''' 5#: BiM
'''

f = open(test_path, encoding='UTF-8')

output_path = output_dir + "\\BiM"
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path += "\\" + domain + "_segment.txt"
output_file = open(output_path, 'w', encoding='UTF-8')

for line in f:
	grams = segmenter.bidirectional_match_segment(line, n)
	write_to_file(grams, output_file)
output_file.close()

#############################################################
f.seek(0)

''' 6#: MPM
'''

f = open(test_path, encoding='UTF-8')

output_path = output_dir + "\\MPM"
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path += "\\" + domain + "_segment.txt"
output_file = open(output_path, 'w', encoding='UTF-8')

for line in f:
	grams = segmenter.most_probability_segment(line, n)
	write_to_file(grams, output_file)
output_file.close()

f.close()