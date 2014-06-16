from utils.files import *
from random import random
import sys

# for now: only unigrams
def generate_n_grams(word, n=None):
	if n is None:
		n = len(word)
	return [word[max(j-n, 0):j] for j in range(len(word)+1)]

class Trie:
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

	def inc(self, key, count=1):
		if not key:
			self.value += count
		if len(key) >= 1:
			if not self.children.has_key(key[0]):
				self.children[key[0]] = Trie()
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
	
def train(input_file, n=None):
	trie = Trie()
	for word, freq in load_tsv_file(input_file, print_progress=True,\
			print_msg='Training %d-gram model...' % n):
#		word = '*' * len(word)
		freq = int(freq)
		for ngr in generate_n_grams(word + '#', n):
			trie.inc(ngr, freq)
#			trie.inc(ngr)
	return trie

#N = 1
#trie = train(INPUT_FILE, N)
#TOTAL = trie[''].value
#trie.normalize()

def word_prob(word, trie, n=None):
	p = 1.0
	for ngr in generate_n_grams(word + '#', n):
		if trie.has_key(ngr):
			p *= trie[ngr].value
		else:
			return 0.0
	return p

def random_word(trie, n=None, context=''):
	if n is None:
		t = trie[context]
	elif n == 1:
		t = trie['']
	else:
		t = trie[context[-n+1:]]
	p = random()
	for c in sorted(t.children.keys()):
		if p <= t.children[c].value:
			if c == '#':
				return ''
			else:
				return c + random_word(trie, n, context+c)
		else:
			p -= t.children[c].value

def generate_random_words(output_file, trie, n=None, num=100000):
	with open_to_write(output_file) as fp:
		i = 0
		while i < num:
			write_line(fp, (random_word(trie, n), ))
			i += 1
	aggregate_file(output_file, key=1)

