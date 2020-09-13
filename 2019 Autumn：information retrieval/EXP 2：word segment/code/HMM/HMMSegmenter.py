import argparse
import os
import codecs

from HMM import Model
from process import Process

def write_to_file(lines, f):
	for line in lines:
		f.write(line)
		f.write('\n')

parser = argparse.ArgumentParser(description='N-gram Chinese word segmentation.')
parser.add_argument('--corpusPath', default='G:\\dataset\\corpus\\WS\\icwb2-data\\training\\pku_training.utf8')
parser.add_argument('--testPath', default='G:\\dataset\\corpus\\WS\\icwb2-data\\testing\\pku_test.utf8')
parser.add_argument('--outputPath', default='G:\\dataset\\corpus\\WS\\result')
p = parser.parse_args()

MAX_WORD_LENGTH = 6

corpus_path = p.corpusPath
test_path = p.testPath
output_path = p.outputPath

print("build model...")

S = ['B', 'E', 'M', 'S']
pro = Process(corpus_path,S)
hidden_states,train=pro._statics()

pro_test = Process(test_path,S)
userless,test = pro_test._statics()

test_wordcount = pro_test._word_count(test)

word_count = pro._word_count(train)

observation = word_count.keys()

conf_prob,trans_prob=pro._tran_conf_prob(train,test_wordcount,word_count,hidden_states)

observations = test
phi = {'B':0.5,'E':0,'M':0,'S':0.5}
model = Model(S,observation,phi,trans_prob,conf_prob)
o_hstate = []

print("begin segment...")

segmenter_name = 'HMM'
domain = corpus_path.split('/')[-1].split('.')[0].split('_')[0]
print("domain: " + domain)

output_path += "\\" + segmenter_name 

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path += "\\" + domain + "_segment.txt"
output_file = open(output_path, 'w', encoding='UTF-8')

for obser in observations:
    length = len(obser)
    index,sub_obser,state= 0,[],[]
    while index < length:
        sub_obser.append(obser[index])
        if obser[index] == '。' or obser[index]=='，':
            sub_state = model.decode(sub_obser)
            sub_obser = []
            state += sub_state
        elif index == length-1:
            sub_state = model.decode(sub_obser)
            sub_obser = []
            state += sub_state
        index += 1
    o_hstate.append(state)

word_sequence = pro._word_sequence(observations,o_hstate)

write_to_file(word_sequence, output_file)

output_file.close()