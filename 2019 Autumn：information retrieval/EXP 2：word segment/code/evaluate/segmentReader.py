class SegmentReader:

	def __init__(self, filename):
		self.f = open(filename, 'r', encoding='UTF-8')

	def reset(self):
		self.f.seek(0)

	def getNextWordAndStems(self):
		line = self.f.readline()
		if not line:
			return None

		segments = line.strip().split()
		return segments

	def close(self):
		self.f.close()