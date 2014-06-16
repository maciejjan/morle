import codecs
import sys

ENCODING = 'utf-8'
INPUT_FILE = sys.argv[1]

#class Trie:
#	def __init__(self):
#		self.count = 0
#		self.children = {}
#	
#	def insert(self, key, count=1):
#		self.count += count
#		if len(key) >= 1:
#			if not self.children.has_key(key[0]):
#				self.children[key[0]] = Trie()
#			self.children[key[0]].insert(key[1:], count)
#	
#	def prob(self, key):
#		if len(key) == 0:
#			return 1.0
#		else:
#			return float(self.children[key[0]].count) / self.count * self.children[key[0]].prob(key[1:])

#def load_wordlist(filename):
#	trie = Trie()
#	with codecs.open(filename, 'r', ENCODING) as fp:
#		for line in fp:
#			split = line.rstrip().split('\t')
#			word, count = split[0], int(split[1])
##			trie.insert(word, count)
#			trie.insert(word)
#	return trie
#
#def train_unigram_model(filename):
#	unigrams = {}
#	total = 0
#	with codecs.open(filename, 'r', ENCODING) as fp:
#		for line in fp:
#			split = line.rstrip().split('\t')
#			word, count = split[0], int(split[1])
#			for c in word:
#				if not unigrams.has_key(c):
#					unigrams[c] = 0
#				unigrams[c] += count
#				total += count
#	for key, val in unigrams.iteritems():
#		unigrams[key] = float(unigrams[key]) / total
#	return unigrams
#
#def unigram_prob(unigrams, word):
#	if len(word) == 0:
#		return 1.0
#	else:
#		return unigrams[word[0]] * unigram_prob(unigrams, word[1:])

words = []
lengths, unigrams = {}, {}
num_words, num_unigrams = 0, 0
with codecs.open(INPUT_FILE, 'r', ENCODING) as fp:
	for line in fp:
		split = line.rstrip().split('\t')
		w_id, word = split[0], split[1]
		words.append((w_id, word))
		l = len(word)
		if not lengths.has_key(l):
			lengths[l] = 0
		lengths[l] += 1
		num_words += 1
		for c in word:
			if not unigrams.has_key(c):
				unigrams[c] = 0
			unigrams[c] += 1
			num_unigrams += 1

for key, val in lengths.iteritems():
	lengths[key] = float(val) / num_words
for key, val in unigrams.iteritems():
	unigrams[key] = float(val) / num_unigrams

for (w_id, word) in words:
	w_prob = lengths[len(word)]
	for c in word:
		w_prob *= unigrams[c]
	w_prob = 1 - (1 - w_prob) ** num_words
	print "UPDATE words SET unigram_prob = {0} WHERE w_id = {1};".format(w_prob, w_id)
