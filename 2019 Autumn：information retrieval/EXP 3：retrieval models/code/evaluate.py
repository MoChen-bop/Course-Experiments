import os
import re
import math
import numpy as np
import argparse
from collections import namedtuple

from corpus.Corpus import Corpus
from corpus.Indexer import Indexer
from corpus.Preprocessor import Preprocessor
from corpus.OHSUMED_Extractor import OHSUMED_Extractor

from model.BooleanModel import BooleanModel
from model.TFModel import TFModel
from model.TF_IDFModel import TF_IDFModel
from model.TS_SSModel import TS_SSModel
from model.LSIModel import LSIModel
from model.WMDModel import WMDModel
from model.LDAModel import LDAModel

def extract_gold_from_file(gold_file_path):
	golds = {}
	GoldItem = namedtuple('GoldItem', ['docID', 'score'])
	f = open(gold_file_path, 'r', encoding='utf-8')

	for line in f:
		try:
			topicID, docID, score = line.strip().split('\t')
		except:
			topicID, docID = line.strip().split('\t')
			score = 1
		docID = int(docID)
		if topicID not in golds.keys():
			golds[topicID] = {}
		golds[topicID][docID] = int(score)

	f.close()
	return golds


def extract_query_from_file(query_file_path):
	query_items = {}
	QueryItem = namedtuple('QueryItem', ['title', 'desc'])
	TOP_PATTERN = re.compile(r'<top>(.*?)<\/top>', re.DOTALL | re.M)
	NUM_PATTERN = re.compile(r'Number:(.*?)<title>', re.DOTALL | re.M)
	TITLE_PATTERN = re.compile(r'<title>(.*?)<desc>', re.DOTALL | re.M)
	DESC_PATTERN = re.compile(r'Description:\n(.*?)\n', re.DOTALL | re.M)

	f = open(query_file_path, 'r', encoding='utf-8')
	lines = f.read()

	for item in TOP_PATTERN.findall(lines):
		topicID = NUM_PATTERN.findall(item)[0].strip()
		title = TITLE_PATTERN.findall(item)[0].strip()
		desc = DESC_PATTERN.findall(item)[0].strip()
		query_item = QueryItem(title, desc)
		query_items[topicID] = query_item

	f.close()
	return query_items


def evaluate_model(model, model_name, max_query_num=60, max_topK=100, reevaluate=False, normalized=False):
	topicIDs = list(gold_items.keys())
	model_output_path = output_path + "\\" + model_name
	if not os.path.exists(model_output_path):
		os.makedirs(model_output_path)
	output_query_result_file_path = model_output_path + "\\query_result.txt"
	output_matrics_file_path = model_output_path + "\\matrics.txt"
	#if not reevaluate and os.path.exists(output_query_result_file_path):
		#return
	output_query_result_file = open(output_query_result_file_path, 'w', encoding='utf-8')
	output_matrics_file = open(output_matrics_file_path, 'w', encoding='utf-8')
	max_query_num = min(len(query_items), len(gold_items), max_query_num)
	for i in range(max_query_num):
		topicID = topicIDs[i]
		gold = gold_items[topicID]
		query = query_items[topicID]

		topK = min(max_topK, len(gold) * 10)
		query_result = model.search(query.desc, topK=topK, normalized=normalized)
		matrics, tags = calculate_matrics(query_result, topicID, gold, query)
		flush_result(topicID, query_result, tags, matrics, 
					 output_query_result_file, output_matrics_file)

	output_query_result_file.close()
	output_matrics_file.close()



def calculate_matrics(query_result, topicID, gold, query):
	Matrics = namedtuple('Matrics', ['P', 'R', 'F1', 'CG', 'DCG', 'nDCG', 'AP'])
	precise_at_k = []
	recall_at_k =[]
	F1_at_k = []
	CG_at_k = []
	DCG_at_k = []
	nDCG_at_k = []
	AP_at_k = []
	tags = []

	TP = 0
	FP = 0
	TN = len(gold)
	gain = 0
	discounted_gain = 0
	ideal_gain = 0
	count = 0

	for _, score in gold.items():
		if score == 2:
			count += 1

	k = 0
	for docID, _ in query_result:
		k += 1

		if k == 1:
			ideal_gain += 2
		elif k <= count:
			ideal_gain += 2 / math.log(k)
		elif k <= len(gold):
			ideal_gain += 1

		if docID in gold.keys():
			TP += 1
			TN -= 1
			gain += gold[docID]
			if k == 1:
				discounted_gain += gold[docID]
			else:
				discounted_gain += gold[docID] / math.log(k)
			tags.append('R')
		else:
			FP += 1
			tags.append('N')

		p = TP / (TP + FP)
		r = TP / (TP + TN)
		if p + r == 0:
			f1 = 0
		else:
			f1 = 2 * p * r / (p + r)
		precise_at_k.append(p)
		recall_at_k.append(r)
		F1_at_k.append(f1)
		CG_at_k.append(gain)
		DCG_at_k.append(discounted_gain)
		nDCG_at_k.append(discounted_gain / ideal_gain)
		AP = np.mean(precise_at_k)
		AP_at_k.append(AP)

	matrics = Matrics(precise_at_k, recall_at_k, F1_at_k, CG_at_k, DCG_at_k, nDCG_at_k, AP_at_k)
	return matrics, tags


def flush_result(topicID, query_result, tags, matrics, 
				 output_query_result_file, output_matrics_file):
	output_query_result_file.write(str(topicID) + ":\n")
	for i, result_item in enumerate(query_result):
		docID, similarity = result_item
		output_query_result_file.write(str(docID) + '\t' +
						 str(similarity) + '\t' + tags[i] + '\n')
	output_query_result_file.write('\n')

	output_matrics_file.write(str(topicID) + ":\n")
	for i in range(len(matrics.P)):
		output_matrics_file.write('%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' % 
								  (matrics.P[i],
								   matrics.R[i],
								   matrics.F1[i],
								   matrics.CG[i],
								   matrics.DCG[i],
								   matrics.nDCG[i],
								   matrics.AP[i]) )
	output_matrics_file.write("\n")








parser = argparse.ArgumentParser(description='evaluate IR model')
parser.add_argument('--test_document_file', required=False, 
				    default='G:\\dataset\\corpus\\OHSUMED\\ohsu-trec\\trec9-test\\ohsumed.88-91',
				    help='Path to the file which contains the documents to be indexed')
parser.add_argument('--output_path', required=False, 
					default='G:\\dataset\\corpus\\OHSUMED\\EvaluateReasult\\mesh-88-91-test')
parser.add_argument('--gold_file_path', required=False, 
					default='G:\\dataset\\corpus\\OHSUMED\\ohsu-trec\\trec9-test\\qrels.mesh.88-91')
parser.add_argument('--query_file_path', required=False, 
					default='G:\\dataset\\corpus\\OHSUMED\\ohsu-trec\\trec9-train\\query.mesh.1-4904')
parser.add_argument('--enable_case_folding', required=False, default=True)
parser.add_argument('--enable_stemmer', required=False, default=False)
parser.add_argument('--enable_lemmatizer', required=False, default=True)
parser.add_argument('--enable_remove_stop_words', required=False, default=True)
p = parser.parse_args()

document_file = p.test_document_file
output_path = p.output_path
gold_file_path = p.gold_file_path
query_file_path = p.query_file_path

gold_items = extract_gold_from_file(gold_file_path)
query_items = extract_query_from_file(query_file_path)

print('begin indexing...')
extractor = OHSUMED_Extractor()
corpus = Corpus(extractor)
corpus.build(document_file)
preprocessor = Preprocessor(enable_case_folding=p.enable_case_folding, 
							enable_remove_stop_words=p.enable_remove_stop_words,
							enable_stemmer=p.enable_stemmer,
							enable_lemmatizer=p.enable_lemmatizer)
output_index_file_path = output_path + '\\index'
indexer = Indexer(preprocessor, 'content', output_index_file_path)
indexer.index(corpus)


print('evaluating boolean model...')
boolean_model = BooleanModel(indexer)
evaluate_model(boolean_model, "Boolean")

print('evaluating tf model...')
tf_model = TFModel(corpus, indexer)
evaluate_model(tf_model, "TF", normalized=False)

print('evaluating tf-idf model...')
args = 'log'
tf_idf_model = TF_IDFModel(corpus, indexer, args)
evaluate_model(tf_idf_model, "TF-IDF", normalized=False)

print('evaluating ts-ss model...')
ts_ss_model = TS_SSModel(corpus, indexer, preprocessor, measure='TS-SS')
evaluate_model(ts_ss_model, "TS-SS")

print('evaluating lsi model...')
lsi_model = LSIModel(corpus, preprocessor, output_path + "\\lsi")
lsi_model.load(regenerate=True)
evaluate_model(lsi_model, "LSI")

print('evaluating lda model...')
lda_model = LDAModel(corpus, preprocessor)
lda_model.build()
evaluate_model(lda_model, "LDA")

#print('evaluating wmd model...')
#wmd_model = WMDModel(corpus, indexer, preprocessor)
#wmd_model.build()
#evaluate_model(wmd_model, "WMD", max_query_num=5)