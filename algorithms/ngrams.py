from datastruct.counter import *
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
		self.total = 0
		self.trie = NGramTrie()
		if settings.USE_TAGS:
			self.tags = Counter()
	
	def train(self, words):
		total = 0
		for word, freq in words:
			if settings.USE_TAGS:
				idx = word.rfind(u'_')
				word, tag = word[:idx], word[idx+1:]
				self.tags.inc(tag, freq)
			for ngr in generate_n_grams(word + '#', self.n):
				self.total += freq
				self.trie.inc(ngr, freq)
		self.trie.normalize()
	
	def train_from_file(self, filename):
		self.train(read_tsv_file(filename, (str, int), print_progress=True,\
			print_msg='Training %d-gram model...' % self.n))
	
	def ngram_prob(self, string):
		p = 1.0
		for ngr in generate_n_grams(string, self.n):
			if ngr in self.trie:
				p *= self.trie[ngr].value
			else:
				p *= 1.0 / self.total
		return p
	
	def word_prob(self, word):
		result = 1.0
		if settings.USE_TAGS:
			idx = word.rfind(u'_')
			word, tag = word[:idx], word[idx+1:]
			if tag in self.tags:
				result *= self.tags[tag] / self.tags.total
			else:
				result *= 1 / self.tags.total	# smoothing
		result *= self.ngram_prob(word + '#')
		return result
	
	def save_to_file(self, filename):	#TODO deprecated
		with open_to_write(filename) as fp:
			write_line(fp, (u'', self.total))
			for ngr, p in self.ngrams.items():
				write_line(fp, (ngr, p))
	
	@staticmethod
	def load_from_file(filename):	#TODO deprecated
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
		elif key[0] in self.children:
			try:
				return self.children[key[0]].__getitem__(key[1:])
			except KeyError:
				raise KeyError(key)
		else:
			raise KeyError(key)
	
	def __contains__(self, key):
		if not key:
			return True
		elif len(key) == 1:
			return key in self.children
		elif len(key) > 1:
			if key[0] in self.children:
				return key[1:] in self.children[key[0]]
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
			if key[0] not in self.children:
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
class TrigramHash:
    def __init__(self):
        self.entries = {}
        self.num_words = 0
        if settings.USE_TAGS:
            self.tag_entries = {}
    
    def add_to_key(self, word, key):
        if key not in self.entries:
            self.entries[key] = set()
        self.entries[key].add(word)
        
    def add(self, word):
        self.num_words += 1
        word_full = word
        if settings.USE_TAGS:
            idx = word.rfind('_')
            word, tag = word[:idx], word[idx+1:]
            if tag not in self.tag_entries:
                self.tag_entries[tag] = set()
            self.tag_entries[tag].add(word_full)

        for tr in generate_n_grams(''.join(['^', word, '$']), 3):
            if len(tr) == 3:
                self.add_to_key(word_full, tr)
                self.add_to_key(word_full, '*' + tr[1:])
                self.add_to_key(word_full, tr[:-1] + '*')
                self.add_to_key(word_full, '*' + tr[1:-1] + '*')
    
    def __len__(self):
        return self.num_words
                
    def retrieve(self, trigram):
        return set(self.entries[trigram]) if trigram in self.entries else set([])
	
    def retrieve_tag(self, tag):
        return set(self.tag_entries[tag])

