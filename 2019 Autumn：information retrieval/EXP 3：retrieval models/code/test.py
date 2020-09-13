import os
import argparse

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

parser = argparse.ArgumentParser(description='test IR model')
parser.add_argument('--document_file', required=False, 
			  default='G:\\dataset\\corpus\\OHSUMED\\ohsu-trec\\trec9-train\\ohsumed.87',
			  help='Path to the file which contains the documents to be indexed')
parser.add_argument('--index_file', required=False, default='index', help='Output filename for index file')
parser.add_argument('--output_path', required=False, default='G:\\dataset\\corpus\\OHSUMED\\output',
			  help='Output dictory for result or temp_file')
parser.add_argument('--index_field', required=False, default='content', help='Which field to index')
parser.add_argument('--enable_case_folding', required=False, default=True,
			  help='Enable case folding during preprocessing')
parser.add_argument('--enable_stemmer', required=False, default=False,
			  help='Enable stemmer during preprocessing')
parser.add_argument('--enable_lemmatizer', required=False, default=True,
			  help='Enable lemmatizer during preprocessing')
parser.add_argument('--enable_remove_stop_words', required=False, default=True,
			  help='Enable removal of stop words during preprocesing')
parser.add_argument('--query', required=False, 
			  default='''
			  Some patients converted from ventricular fibrillation to organized
			  rhythms by defibrillation-trained ambulance technicians (EMT-Ds) will
			  refibrillate before hospital arrival. The authors analyzed 271 cases 
			   of ventricular fibrillation managed by EMT-Ds working without paramedic
			    back-up. Of 111 patients initially converted to organized rhythms, 19 (17%) 
			    refibrillated, 11 (58%) of whom were reconverted to perfusing rhythms, including
			     nine of 11 (82%) who had spontaneous pulses prior to refibrillation. 
			     Among patients initially converted to organized rhythms, hospital admission 
			     rates were lower for patients who refibrillated than for patients who did
			      not (53% versus 76%, P = NS), although discharge rates were virtually 
			     identical (37% and 35%, respectively). Scene-to-hospital transport times 
			   were not predictively associated with either the frequency of refibrillation 
			   or patient outcome. Defibrillation-trained EMTs can effectively manage refibrillation 
			   with additional shocks and are not at a significant disadvantage when paramedic back-up 
			   is not available.''', help='Query')

p = parser.parse_args()

extractor = OHSUMED_Extractor()
corpus = Corpus(extractor)
print('building corpus')
corpus.build(p.document_file)
preprocessor = Preprocessor(enable_case_folding=p.enable_case_folding, 
							enable_remove_stop_words=p.enable_remove_stop_words,
							enable_stemmer=p.enable_stemmer,
							enable_lemmatizer=p.enable_lemmatizer)
output_index_file_path = p.output_path + '\\index'
print('begin indexing...')
indexer = Indexer(preprocessor, p.index_field, output_index_file_path)
indexer.index(corpus)
print('test boolean model')
boolean_model = BooleanModel(indexer)
print(boolean_model.search(p.query, normalized=True))

print('test tf model')
tf_model = TFModel(corpus, indexer)
print(tf_model.search(p.query, normalized=True))

print('test tf-idf model')
args = ('add', 0.3)
tf_idf_model = TF_IDFModel(corpus, indexer, args)
print(tf_idf_model.search(p.query, normalized=True))

print('test ts-ss model')
ts_ss_model = TS_SSModel(corpus, indexer, preprocessor, measure='TS-SS')
print(ts_ss_model.search(p.query))

print("test lsi model")
lsi_model = LSIModel(corpus, preprocessor, p.output_path)
lsi_model.load()
print(lsi_model.search(p.query))

print("test lda model")
lda_model = LDAModel(corpus, preprocessor)
lda_model.build()
print(lda_model.search(p.query))

print("test wmd model")
wmd_model = WMDModel(corpus, indexer, preprocessor)
wmd_model.build()
print(wmd_model.search(p.query))
