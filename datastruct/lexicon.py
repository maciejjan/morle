from datastruct.counter import *
import algorithms.ngrams
from utils.files import *
import math

class LexiconNode:
	def __init__(self, word, freq, sigma, ngram_prob, corpus_prob, sum_weights):
		self.word = word
		self.freq = freq
		self.sigma = sigma
		self.ngram_prob = ngram_prob
		self.corpus_prob = corpus_prob
		self.sum_weights = sum_weights
		self.prev = None
		self.next = {}

	def root(self):
		root = self
		while root.prev is not None:
			root = root.prev
		return root
	
	def analysis(self):
		analysis = []
		node = self
		while node.prev is not None:
			node = node.prev
			analysis.append(node.word)
		return analysis
	
	def forward_multiply_corpus_prob(self, p):
		self.corpus_prob *= p
		for child in self.next.values():
			child.forward_multiply_corpus_prob(p)
	
	def backward_add_sigma(self, s):
		self.sigma += s
		if self.prev is not None:
			self.prev.backward_add_sigma(s)

class Lexicon:
	def __init__(self):
		self.nodes = {}
		self.roots = set([])
		self.rules_c = Counter()
		self.rules_freq = Counter()
		self.total = 0
#		self.sigma_total = self.total
	
	def __len__(self):
		return len(self.nodes)
	
	def has_key(self, key):
		return self.nodes.has_key(key)
	
	def keys(self):
		return self.nodes.keys()

	def values(self):
		return self.nodes.values()
	
	def __getitem__(self, key):
		return self.nodes[key]
	
	def __setitem__(self, key, val):
		self.nodes[key] = val
	
	def lexicon_logl(self, rules):
		logl = 0.0
		# lexicon log-likelihood given rules
		logl += sum([math.log(i) for i in range(1, len(self.roots)+1)])
		for root_w in self.roots:
			logl += math.log(self.nodes[root_w].ngram_prob)
		for r, count in self.rules_c.iteritems():
			m = rules[r].domsize - count
			logl += count * math.log(rules[r].prod) +\
				m * math.log(1.0 - rules[r].prod)
		return logl
	
	def corpus_logl(self, rules):
		logl = 0.0
		# corpus logl-likelihood given lexicon and rules
		for root_w in self.roots:
			root = self.nodes[root_w]
			logl += root.sigma * (math.log(root.sigma) - math.log(self.total))
		for word in self.values():
			logl -= word.sigma * math.log(word.sum_weights)
		for rule, freq in self.rules_freq.iteritems():
			logl += freq * math.log(rules[rule].weight)
		logl += self.total * math.log(rules[u'#'].weight)
		return logl

	def logl(self, rules):
		return self.lexicon_logl(rules) + self.corpus_logl(rules)
	
	def logl_gradient(self, rules):
		d = {}
		for r in self.rules_freq.keys():
			d[r] = float(self.rules_freq[r]) / rules[r].weight
		d[u'#'] = float(self.total) / rules[u'#'].weight
		for word in self.values():
			for r in word.next.keys():
				d[r] -= float(word.sigma) / word.sum_weights
			d[u'#'] -= float(word.freq) / word.sum_weights
		return d
	
	def try_weights(self, rules_w):
		logl = 0.0
		for root_w in self.roots:
			root = self.nodes[root_w]
			logl += root.sigma * (math.log(root.sigma) - math.log(self.total))
		for word in self.values():
			logl -= word.sigma * math.log(rules_w[u'#'] +\
				sum([rules_w[r] for r in word.next.keys()]))
		for rule, freq in self.rules_freq.iteritems():
			logl += freq * math.log(rules_w[rule])
		logl += self.total * math.log(rules_w[u'#'])
		return logl
	
	def try_edge(self, word_1, word_2, rule):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		root = w1.root()
		# change of corpus log-likelihood
		result = root.sigma * (math.log(root.sigma + w2.sigma) - math.log(root.sigma))
		result += w2.sigma * (math.log(root.sigma + w2.sigma) - math.log(w2.sigma))
		result += w2.sigma * (math.log(w1.corpus_prob * w1.sum_weights * rule.weight) -\
			math.log(root.corpus_prob * root.sum_weights * (w1.sum_weights + rule.weight)))
		result += w1.sigma * (math.log(w1.sum_weights) - math.log(w1.sum_weights + rule.weight))
		# change of lexicon log-likelihood
		result += math.log(rule.prod) - math.log(len(self.roots) * w2.ngram_prob * (1.0 - rule.prod))
		return result
	
	def draw_edge(self, word_1, word_2, rule):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		root = w1.root()

		# update word probabilities
		w2.forward_multiply_corpus_prob(w1.corpus_prob * w1.sum_weights * rule.weight /\
			(root.corpus_prob * root.sum_weights * (w1.sum_weights + rule.weight)))
		root.forward_multiply_corpus_prob(float(root.sigma + w2.sigma) / root.sigma)
		w2.forward_multiply_corpus_prob(float(root.sigma + w2.sigma) / w2.sigma)
		w1.forward_multiply_corpus_prob(float(w1.sum_weights) / (w1.sum_weights + rule.weight))

		# update frequency and weight sums
		w1.backward_add_sigma(w2.sigma)
		w1.sum_weights += rule.weight

		# draw the edge
		w2.prev = w1
		w1.next[rule.rule] = w2

		# update global information
		self.roots.remove(word_2)
		self.rules_c.inc(rule.rule)
		self.rules_freq.inc(rule.rule, w2.sigma)

		# update rule frequencies on the path
		n = w1
		while n.prev is not None:
			for r, n2 in n.prev.next.iteritems():
				if n2 == n:
					self.rules_freq.inc(r, w2.sigma)
					break
			n = n.prev

#		self.sigma_total += w2.sigma * (len(w1.analysis()) + 1)
#		if self.sigma_total != self.rules_freq.total + self.total:
#			raise Exception('ASSERTION FAILED: %s, %s, %s, %d = %d, %d != %d, %s, %d, %d, %d' %\
#				(word_1, word_2, rule.rule, before_1, before_2,\
#				self.sigma_total, self.rules_freq.total + self.total, w1.analysis(), w1.sigma, w2.sigma, c))
	
	# remove all edges
	def reset(self):
		self.rules_c = Counter()
		self.rules_freq = Counter()
		for n in self.nodes.values():
			n.prev = None
			n.next = {}
			n.sigma = n.freq
			n.corpus_prob = float(n.freq) / self.total
			n.sum_weights = 1.0
			self.roots.add(n.word)
	
	def save_to_file(self, filename):
		def write_subtree(fp, word_1, word_2, rule):
			w2 = self.nodes[word_2]
			write_line(fp, (word_1, word_2, rule, w2.freq, w2.sigma, w2.ngram_prob, w2.corpus_prob, w2.sum_weights))
			for next_rule, next_word in w2.next.iteritems():
				write_subtree(fp, word_2, next_word.word, next_rule)
		with open_to_write(filename) as fp:
			for rt in self.roots:
				write_subtree(fp, u'', rt, u'')

	@staticmethod
	def init_from_file(filename):
		lexicon = Lexicon()
		unigrams = algorithms.ngrams.NGramModel(1)
		unigrams.train_from_file(filename)
		for word, freq in read_tsv_file(filename, (unicode, int)):
			lexicon[word] = LexiconNode(word, freq, freq, unigrams.word_prob(word), 0.0, 1.0)
			lexicon.roots.add(word)
			lexicon.total += freq
#			lexicon.sigma_total += freq
		# compute corpus probabilities
		for word in lexicon.values():
			word.corpus_prob = float(word.freq) / lexicon.total
		return lexicon
	
	@staticmethod
	def load_from_file(filename):
		lexicon = Lexicon()
		for word_1, word_2, rule, freq, sigma, ngram_prob, corpus_prob, sum_weights in read_tsv_file(\
				filename, types=(unicode, unicode, unicode, int, int, float, float, float)):
			lexicon[word_2] = LexiconNode(word_2, freq, sigma, ngram_prob, corpus_prob, sum_weights)
			lexicon.total += freq
			if word_1 and rule:
				lexicon[word_2].prev = word_1
				lexicon[word_1].next[rule] = word_2
				lexicon.rules_c.inc(rule)
				lexicon.rules_freq.inc(rule, sigma)
			else:
				lexicon.roots.add(word_2)
		return lexicon
	
