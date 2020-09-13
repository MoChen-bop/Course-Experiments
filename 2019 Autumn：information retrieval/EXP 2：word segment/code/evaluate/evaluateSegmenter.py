import os
import argparse
import re

from accounter import Accounter
from segmentReader import SegmentReader
import edit_path

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def extract_lexicon(corpus_file_path):
	lexicon = {}
	f = open(corpus_file_path, 'r', encoding='UTF-8')
	for line in f:
		line = line.strip().split()
		for word in line:
			if word in lexicon.keys():
				lexicon[word] += 1
			else:
				lexicon[word] = 1
	return lexicon

def evaluate(segment_file_path, gold_file_path, corpus_file_path, output_directory):
	segment_reader = SegmentReader(segment_file_path)
	gold_reader = SegmentReader(gold_file_path)

	segment_accounter = Accounter()
	gold_accounter = Accounter()

	sentID = 0

	TP = 0
	FN = 0
	FP = 0

	OOV = 0
	OOV_index = 0

	total_count = 0
	OSI = 0
	USI = 0
	WSI = 0

	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	result_file = open(output_directory + '\\evaluate_matrics.txt', 'w', encoding='UTF-8')
	oov_file = open(output_directory + '\\oov.txt', 'w', encoding='UTF-8')
	over_segmenting_file = open(output_directory + '\\over_sementing.txt', 'w', encoding='UTF-8')
	under_segmenting_file = open(output_directory + '\\under_segmenting.txt', 'w', encoding='UTF-8')
	wrong_segmenting_file = open(output_directory + '\\wrong_segmenting.txt', 'w', encoding='UTF-8')
	low_precise_file = open(output_directory + '\\low_precise.txt', 'w', encoding='UTF-8')
	low_recall_file = open(output_directory + '\\low_recall.txt', 'w', encoding='UTF-8')

	lexicon = extract_lexicon(corpus_file_path)

	print("Stage #1...")
	while True:
		segments = segment_reader.getNextWordAndStems()
		golds = gold_reader.getNextWordAndStems()

		if segments == None or golds == None:
			break

		for word in segments:
			segment_accounter.addHit(word, sentID)
		for gold in golds:
			gold_accounter.addHit(gold, sentID)

			if is_all_chinese(gold) and gold not in lexicon.keys():
				OOV += 1
				if gold in segments:
					OOV_index += 1
					oov_file.write(gold + '\n')
				else:
					oov_file.write("### not segmented ### " + gold + "\n")

		sentID += 1

		I_count, D_count, M_count = edit_path.compare(segments, golds)

		if not len(golds) == 0 and I_count / len(golds) > 0.4:
			under_segmenting_file.write("Golds: " + str(golds) + "\nSegments: " + str(segments) + "\n\n")
		if not len(golds) == 0 and D_count / len(golds) > 0.4:
			over_segmenting_file.write("Golds: " + str(golds) + "\nSegments: " + str(segments) + "\n\n")
		if not len(golds) == 0 and M_count / len(golds) > 0.4:
			wrong_segmenting_file.write("Golds: " + str(golds) + "\nSegments: " + str(segments) + "\n\n")

		OSI += I_count
		USI += D_count
		WSI += M_count

		total_count += len(golds)

	print("Stage #2...")
	segment_reader.reset()
	gold_reader.reset()
	while True:
		golds = gold_reader.getNextWordAndStems()

		if golds == None:
			break

		for gold in golds:
			if not is_all_chinese(gold):
				continue
			gold_sentence_ids = gold_accounter.get_sentence_ids(gold)
			curr_sentence_ids = segment_accounter.get_sentence_ids(gold)

			TP += len(gold_sentence_ids.intersection(curr_sentence_ids))
			FP += len(curr_sentence_ids.difference(gold_sentence_ids))
			FN += len(gold_sentence_ids.difference(curr_sentence_ids))

			if not len(gold_sentence_ids) == 0 and (len(curr_sentence_ids) - len(gold_sentence_ids)) / len(gold_sentence_ids) > 1:
				low_precise_file.write("Gold: " + gold + "\ngold_sentence_ids: " + str(gold_sentence_ids) 
					+ "\ncurr_sentence_ids: " + str(curr_sentence_ids) + "\n\n")
			if not len(curr_sentence_ids) == 0 and (len(gold_sentence_ids) - len(curr_sentence_ids)) / len(curr_sentence_ids) > 1:
				low_recall_file.write("Gold: " + gold + "\ngold_sentence_ids: " + str(gold_sentence_ids) 
					+ "\ncurr_sentence_ids: " + str(curr_sentence_ids) + "\n\n")

	result_file.write("Matrics #1: \n")
	result_file.write("TP = " + str(TP) + ",\tFP = " + str(FP) + ",\tFN = " + str(FN) + "\n")
	p = TP / (TP + FP)  
	r = TP / (TP + FN)
	f1 = 2 * p * r / (p + r)
	result_file.write("Precise: " + str(p) + ',\tRecall: ' + str(r) + "\n")
	result_file.write("F1: " + str(f1))

	result_file.write("\n\nMatrics #2: \n")
	result_file.write("Test Lexicon's size: " + str(len(gold_accounter.getWords())) + "\n")
	if OOV == 0:
		OOV_index = 0
	else:
		OOV_index = OOV_index / OOV
	OOV = OOV / len(gold_accounter.getWords())
	result_file.write("\nOOVI: " + str(OOV_index) + "\n")
	result_file.write("OOV: " + str(OOV) + "\n")

	result_file.write("\nI_count: " + str(OSI) + "\n")
	result_file.write("D_count: " + str(USI) + "\n")
	result_file.write("M_count: " + str(WSI) + "\n")
	result_file.write("Total_count: " + str(total_count) + "\n\n")
	result_file.write("OST: " + str(OSI / total_count) + "\n")
	result_file.write("UST: " + str(USI / total_count) + "\n")
	result_file.write("WST: " + str(WSI / total_count) + "\n")


	segment_reader.close()
	gold_reader.close()
	result_file.close()
	oov_file.close()
	wrong_segmenting_file.close()
	over_segmenting_file.close()
	under_segmenting_file.close()
	low_precise_file.close()
	low_recall_file.close()



parser = argparse.ArgumentParser(description='evaluate segmenter with corpus.')
parser.add_argument('--corpusDirectory', default='G:\\dataset\\corpus\\WS\\icwb2-data\\training')
parser.add_argument('--goldFileDirectory', default='G:\\dataset\\corpus\\WS\\icwb2-data\\gold')
parser.add_argument('--segmentFileDirectory', default='G:\\dataset\\corpus\\WS\\result')
parser.add_argument('--resultPath', default='G:\\dataset\\corpus\\WS\\result\\_evaluate_result')
p = parser.parse_args()

corpus_file_directory = p.corpusDirectory
gold_file_directory = p.goldFileDirectory
segment_file_directory = p.segmentFileDirectory
result_path = p.resultPath

corpus_file_rexp = '.*training.utf8'
gold_file_rexp = '.*test_gold.utf8$'
segment_file_rexp = '.*segment.txt$'

corpus_files = {}
for file_name in os.listdir(corpus_file_directory):
	if re.match(corpus_file_rexp, file_name):
		domain = file_name.split('_')[0]
		corpus_files[domain] = os.path.join(corpus_file_directory, file_name)

gold_files = {}
for file_name in os.listdir(gold_file_directory):
	if re.match(gold_file_rexp, file_name):
		domain = file_name.split('_')[0]
		gold_files[domain] = os.path.join(gold_file_directory, file_name)

segment_file_info = set()
for root, dirs, files in os.walk(segment_file_directory):
	for file in files:
		if re.match(segment_file_rexp, file):
			domain = file.split('_')[0]
			file_path = os.path.join(root, file)
			gold_file_path = gold_files[domain]
			corpus_file_path = corpus_files[domain]
			segmenter_name = root.split("\\")[-1]
			assert segmenter_name is not None
			output_path = result_path + "\\" + domain + "\\" + segmenter_name
			segment_file_info.add((file_path, gold_file_path, corpus_file_path, output_path))

for segment_file_path, gold_file_path, corpus_file_path, output_directory in segment_file_info:
	print(segment_file_path)
	print(gold_file_path)
	print(corpus_file_path)
	print(output_directory)
	print()
	print()
	evaluate(segment_file_path, gold_file_path, corpus_file_path, output_directory)