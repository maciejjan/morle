from utils.files import *
import settings
import random

def generate_n_grams(word, n=None):
	if n is None:
		n = len(word)
	return [word[max(j-n, 0):j] for j in range(len(word)+1) if word[max(j-n, 0):j]]

class NGramModel:
	def __init__(self, n):
		self.n = n
		self.ngrams = {}
		self.total = 0
#		self.trie = NGramTrie()
	
	def train(self, words):
		total = 0
		for word, freq in words:
			if settings.USE_TAGS:
				word = word[:word.rfind(u'_')]
			for ngr in generate_n_grams(word + '#', self.n):
				if not self.ngrams.has_key(ngr):
					self.ngrams[ngr] = 0
				self.ngrams[ngr] += freq
				self.total += freq
#				self.trie.inc(ngr, freq)
		for ngr, count in self.ngrams.iteritems():
			self.ngrams[ngr] = float(count) / self.total
#		self.trie.normalize()
	
	def train_from_file(self, filename):
		self.train(read_tsv_file(filename, (unicode, int), print_progress=True,\
			print_msg='Training %d-gram model...' % self.n))
	
	def word_prob(self, word):
		p = 1.0
		if settings.USE_TAGS:
			word = word[:word.rfind(u'_')]
		for ngr in generate_n_grams(word + '#', self.n):
			if self.ngrams.has_key(ngr):
				p *= self.ngrams[ngr]
#				p *= self.trie[ngr].value
			else:
				p *= 1.0 / self.total
#				return 0.0
		return p
	
#	def random_word(self):
#		return self.trie.random_word()
	
	def save_to_file(self, filename):
		with open_to_write(filename) as fp:
			write_line(fp, (u'', self.total))
			for ngr, p in self.ngrams.iteritems():
				write_line(fp, (ngr, p))
	
	@staticmethod
	def load_from_file(filename):
		model = NGramModel(None)
		for ngr, p in read_tsv_file(filename, (unicode, float)):
			if ngr == u'':
				model.total = int(p)
			else:
				model.ngrams[ngr] = p
		model.n = max([len(ngr) for ngr in model.ngrams.keys()])
		return model

class NGramTrie:
	def __init__(self):
		self.value = 0
		self.children = {}
	
	def __getitem__(self, key):
		if not key:
			if self.value is not None:
				return self
			else:
				raise KeyError(key)
		elif self.children.has_key(key[0]):
			try:
				return self.children[key[0]].__getitem__(key[1:])
			except KeyError:
				raise KeyError(key)
		else:
			raise KeyError(key)
	
	def has_key(self, key):
		if not key:
			return True
		elif len(key) == 1:
			return self.children.has_key(key)
		elif len(key) > 1:
			if self.children.has_key(key[0]):
				return self.children[key[0]].has_key(key[1:])
			else:	
				return False
		else:
			return False
	
	def keys(self):
		pass

	def inc(self, key, count=1):
		if not key:
			self.value += count
		if len(key) >= 1:
			if not self.children.has_key(key[0]):
				self.children[key[0]] = NGramTrie()
			self.children[key[0]].inc(key[1:], count)

	def __len__(self):
		raise Exception('Not implemented yet.')
	
	def normalize(self, parent_count=None):
		my_count = sum([c.value for c in self.children.values()])
		for c in self.children.values():
			c.normalize(my_count)
		if parent_count is None:
			self.value = 1.0
		else:
			if parent_count > 0:
				self.value = float(self.value) / parent_count
			elif self.value > 0:
				raise Exception('can\'t be!')
			if self.value > 1.0:
				raise Exception('boom!')
	
	def random_word(self):
		p = random()
		for c in sorted(self.children.keys()):
			if p <= self.children[c].value:
				if c == '#':
					return ''
				else:
					if self.children[c].children:
						return c + self.children[c].random_word()
			else:
				p -= self.children[c].value

# use to quickly retrieve all words matching the left side of a rule
class WordNGramHash:
	def __init__(self, n):
		self.n = n
