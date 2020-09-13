import re
import os

stem_output_path = "G:\\dataset\\corpus\\BNC\\Stem\\NOSTEM"
word_path = "G:\\dataset\\corpus\\BNC\\Corpus"

word_file_rexp = ".*.bnc.words.txt$"

for file_name in os.listdir(word_path):
	if re.match(word_file_rexp, file_name):
		domain = file_name.split('.')[0]
		output_file_path = stem_output_path + "\\" + domain + ".bnc.stem.txt"
		if not os.path.exists(stem_output_path):
			os.makedirs(stem_output_path)
		output_file = open(output_file_path, 'w', encoding='utf-8')
		word_file = open(os.path.join(word_path, file_name), 'r', encoding='utf-8')
		for line in word_file:
			word = line.strip()
			output_file.write(word + '\t' + word + '\n')
		output_file.close()
		word_file.close()
