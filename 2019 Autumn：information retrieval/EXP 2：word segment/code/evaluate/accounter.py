class Accounter:

	def __init__(self):
		self.words = []
		self.words2sentence_ids = {}


	def addHit(self, word, sentence_id):
		self.words.append(word)

		i = self.words2sentence_ids.get(word, None)
		if i == None:
			self.words2sentence_ids[word] = set([sentence_id])
		elif sentence_id not in i:
			i.add(sentence_id)


	def getWords(self):
		return self.words


	def get_sentence_ids(self, word):
		ids = self.words2sentence_ids.get(word, None)
		if ids == None:
			return set()
		else:
			return ids


	def print(self):
		print('\nWords: ' + str(self.words))
		print('\nIds: ' + str(self.words2sentence_ids))