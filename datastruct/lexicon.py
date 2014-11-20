from datastruct.counter import *
from datastruct.rules import *
from algorithms.align import lcs
import algorithms.ngrams
from utils.files import *
import settings
import math

def align_words(word_1, word_2):
	cs = algorithms.align.lcs(word_1, word_2)
	pattern = re.compile('(.*)' + '(.*?)'.join([\
		letter for letter in cs]) + '(.*)')
	m1 = pattern.search(word_1)
	m2 = pattern.search(word_2)
	alignment = []
	for i, (x, y) in enumerate(zip(m1.groups(), m2.groups())):
		if x or y:
			alignment.append((x, y))
		if i < len(cs):
			alignment.append((cs[i], cs[i]))
	return alignment


class LexiconNode:
	def __init__(self, word, freq, sigma, ngram_prob, corpus_prob, sum_weights):
		self.word = word
#		self.stem = word
		self.freq = freq
		self.sigma = sigma
		self.ngram_prob = ngram_prob
		self.corpus_prob = corpus_prob
		self.sum_weights = sum_weights
		self.prev = None
		self.next = {}
		self.training = True
		self.structure = None

	def root(self):
		root = self
		while root.prev is not None:
			root = root.prev
		return root
	
	def depth(self):
		return len(self.analysis())
	
	def analysis(self):
		analysis = []
		node = self
		while node.prev is not None:
			node = node.prev
			analysis.append(node.word)
		return analysis

	def analysis_comp(self):
		analysis = []
		node = self
		while node.prev is not None:
			word = node.prev.word
			rule = [r for r, n in node.prev.next.iteritems() if n == node][0]
			rule_sp = rule.split('*')
			if len(rule_sp) == 3:
				if rule_sp[0].find('/') > -1:
					word = word + '+(' + rule_sp[1] + ')'
				elif rule_sp[2].find('/') > -1:
					word = '(' + rule_sp[1] + ')+' + word
#			if not node.prev.prev or len(lcs(node.word, node.prev.word)) > len(lcs(node.word, node.prev.prev.word)):
			analysis.append(word)
			node = node.prev
		return analysis

#	def analysis_stems(self):
#		analysis = []
#		node = self
#		analysis.append(node.stem)
#		while node.prev is not None:
#			word = node.prev.stem
#			rule = [r for r, n in node.prev.next.iteritems() if n == node][0]
#			rule_sp = rule.split('*')
#			if len(rule_sp) == 3:
#				if rule_sp[0].find('/') > -1:
#					word = word + '+(' + rule_sp[1] + ')'
#				elif rule_sp[2].find('/') > -1:
#					word = '(' + rule_sp[1] + ')+' + word
##			if not node.prev.prev or len(lcs(node.word, node.prev.word)) > len(lcs(node.word, node.prev.prev.word)):
#			analysis.append(word)
#			node = node.prev
#		return analysis
	
	def analysis_morphochal(self):
		analysis = []
		node = self
		while node.prev is not None:
			word = node.prev.word
			rule = [r for r, n in node.prev.next.iteritems() if n == node][0]
			rule_sp = rule.split('*')
			if len(rule_sp) == 3:
				rule = rule_sp[0] + '*' + rule_sp[2]
#				rule = rule_sp[0] + rule_sp[2]
				analysis.append(rule_sp[1])
			else:
				analysis.append(rule)
#			rule = Rule.from_string(rule)
#			if rule.prefix[0]:
#				analysis.append(rule.prefix[0] + '-')
#			if rule.prefix[1]:
#				analysis.append(rule.prefix[1] + '+')
#			if rule.suffix[0]:
#				analysis.append('-' + rule.suffix[0])
#			if rule.suffix[1]:
#				analysis.append('+' + rule.suffix[1])
#			for x, y in rule.alternations:
#				if x:
#					analysis.append('-' + x + '-')
#				if y:
#					analysis.append('+' + y + '+')
			node = node.prev
		analysis.append(node.word)
		to_remove = []
		for x in analysis:
			if x.find('-') > -1 and x.replace('-', '+') in analysis:
				to_remove.append(x)
				to_remove.append(x.replace('-', '+'))
			elif x.find('-') > -1 and x.replace('-', '') in analysis:
				to_remove.append(x)
				to_remove.append(x.replace('-', ''))
		for x in to_remove:
			if x in analysis:
				analysis.remove(x)
		analysis = [x for x in analysis \
			if (not '+' in x and not '-' in x)\
				or (x.replace('+', '') in self.word)]
		analysis.reverse()
		return analysis

	def show_tree(self, space=''):
		print space + self.word.encode('utf-8'), self.freq #, self.sigma
		for w in self.next.values():
			w.show_tree(space=space+'\t')
	
	def words_in_tree(self):
		result = [self.word]
		for w in self.next.values():
			result.extend(w.words_in_tree())
		return result
	
	def forward_multiply_corpus_prob(self, p):
		self.corpus_prob *= p
		for child in self.next.values():
			child.forward_multiply_corpus_prob(p)
	
	def backward_add_sigma(self, s):
		self.sigma += s
		if self.prev is not None:
			self.prev.backward_add_sigma(s)
	
#	def backward_update_stem(self):
#		if self.prev is not None:
#			self.prev.stem = lcs(self.prev.stem, self.stem)
#			self.prev.backward_update_stem()

	def annotate_word_structure(self, depth=0):
		self.structure = [depth] * len(self.word)
		node = self
		node_depth = depth
		while node.prev is not None:
			node = node.prev
			node_depth -= 1
			alignment = align_words(self.word, node.word)
			i = 0
			for x, y in alignment:
				if x == y:
					self.structure[i] = node_depth
					i += 1
				else:
					i += len(x)
		# fix prefixes
		for i in range(len(self.structure)-1, 0, -1):
			if self.structure[i-1] < self.structure[i] and not 0 in self.structure[:i]:
				self.structure[i-1] = self.structure[i]
		# fix suffixes
		for i in range(len(self.structure)-1):
			if self.structure[i+1] < self.structure[i] and not 0 in self.structure[i:]:
				self.structure[i+1] = self.structure[i]
		for child in self.next.values():
			child.annotate_word_structure(depth+1)
	
	def split(self):
		split = []
		cur_morph = self.word[0]
		for i in range(1, len(self.structure)):
			if self.structure[i] == self.structure[i-1]:
				cur_morph += self.word[i]
			else:
				split.append(cur_morph)
				cur_morph = self.word[i]
		split.append(cur_morph)
		return split

	def show_split_tree(self, space=''):
		print space + '|'.join(self.split()).encode('utf-8'), self.freq #, self.sigma
		for w in self.next.values():
			w.show_split_tree(space=space+'\t')

class Lexicon:
	def __init__(self):
		self.nodes = {}
		self.roots = set([])
		self.rules_c = Counter()
		self.rules_freq = Counter()
		self.total = 0
		self.sigma_total = self.total
	
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
	
	def add_word(self, word, freq, ngram_prob):
		self.nodes[word] = LexiconNode(word, freq, freq, ngram_prob, 0.0, 1.0)
		self.total += freq
		self.nodes[word].corpus_prob = float(freq) / self.total
		for rt in self.roots:
			self.nodes[rt].forward_multiply_corpus_prob(float(self.total-freq) / self.total)
		self.roots.add(word)
	
	def lexicon_logl(self, rules):
		logl = 0.0
		# lexicon log-likelihood given rules
		logl += sum([math.log(i) for i in range(1, len(self.roots)+1)])
		for root_w in self.roots:
			logl += math.log(self.nodes[root_w].ngram_prob)
		for r, count in self.rules_c.iteritems():
			m = rules[r].domsize - count
			logl += count * math.log(rules[r].prod)
			if m > 0:
				logl += m * math.log(1.0 - rules[r].prod)
		return logl
	
	def corpus_logl_old(self, rules):
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
	
	def corpus_logl(self, rules):
		logl = 0.0
		for w1 in self.values():
			for rule, w2 in w1.next.iteritems():
				logl += math.log(rules[rule].freqprob(w2.freq - w1.freq))
		return logl

	def logl(self, rules):
		if settings.USE_WORD_FREQ:
			return self.lexicon_logl(rules) + self.corpus_logl(rules)
		else:
			return self.lexicon_logl(rules)
	
	def logl_gradient(self, rules):
		d = {}
		for r in self.rules_freq.keys():
			d[r] = float(self.rules_freq[r]) / rules[r].weight
		d[u'#'] = float(self.total) / rules[u'#'].weight
		for word in self.values():
			for r in word.next.keys():
				d[r] -= float(word.sigma) / word.sum_weights
			d[u'#'] -= float(word.sigma) / word.sum_weights
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
	
	def try_edge_pr(self, word_1, word_2, rule):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		root = w1.root()
		print word_1.encode('utf-8'), w1.sigma, w1.corpus_prob, w1.sum_weights
		print word_2.encode('utf-8'), w2.sigma, w2.corpus_prob, w2.sum_weights
		print root.word.encode('utf-8'), root.sigma, root.corpus_prob, root.sum_weights
		print rule.weight
		# change of corpus log-likelihood
		result = root.sigma * (math.log(root.sigma + w2.sigma) - math.log(root.sigma))
		print result
		result += w2.sigma * (math.log(root.sigma + w2.sigma) - math.log(w2.sigma))
		print result
		result += w2.sigma * (math.log(w1.corpus_prob * w1.sum_weights * rule.weight) -\
			math.log(root.corpus_prob * root.sum_weights * (w1.sum_weights + rule.weight)))
		print result
		result += w1.sigma * (math.log(w1.sum_weights) - math.log(w1.sum_weights + rule.weight))
		print result
		# change of lexicon log-likelihood
		result += math.log(rule.prod) - math.log(len(self.roots) * w2.ngram_prob * (1.0 - rule.prod))
		print result
		print ''
		return result
	
	def try_edge(self, word_1, word_2, rule):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		root = w1.root()
		# change of lexicon log-likelihood
		result = math.log(rule.prod) - math.log(len(self.roots) * w2.ngram_prob * (1.0 - rule.prod))
		# change of corpus log-likelihood
#		if settings.USE_WORD_FREQ:
#			result += root.sigma * (math.log(root.sigma + w2.sigma) - math.log(root.sigma))
#			result += w2.sigma * (math.log(root.sigma + w2.sigma) - math.log(w2.sigma))
#			result += w2.sigma * (math.log(w1.corpus_prob * w1.sum_weights * rule.weight) -\
#				math.log(root.corpus_prob * root.sum_weights * (w1.sum_weights + rule.weight)))
#			result += w1.sigma * (math.log(w1.sum_weights) - math.log(w1.sum_weights + rule.weight))
		if settings.USE_WORD_FREQ:
			result += math.log(rule.freqprob(w2.freq - w1.freq))
		return result
	
	def draw_edge(self, word_1, word_2, rule, corpus_prob=True):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		if w1.prev is not None and w1.prev.word == word_2:
			print word_1, word_2
			raise Exception('Cycle detected: %s, %s' % (word_1, word_2))
		root = w1.root()

		# update word probabilities
#		if settings.USE_WORD_FREQ and corpus_prob:
#			w2.forward_multiply_corpus_prob(w1.corpus_prob * w1.sum_weights * rule.weight /\
#				(root.corpus_prob * root.sum_weights * (w1.sum_weights + rule.weight)))
#			root.forward_multiply_corpus_prob(float(root.sigma + w2.sigma) / root.sigma)
#			w2.forward_multiply_corpus_prob(float(root.sigma + w2.sigma) / w2.sigma)
#			w1.forward_multiply_corpus_prob(float(w1.sum_weights) / (w1.sum_weights + rule.weight))
#
#		# update frequency and weight sums
#		w1.backward_add_sigma(w2.sigma)
#		w1.sum_weights += rule.weight
#		self.sigma_total += w2.sigma * (len(w1.analysis()) + 1)

		# draw the edge
		w2.prev = w1
		w1.next[rule.rule] = w2

#		w1.stem = lcs(w1.stem, lcs(w1.word, w2.word))
#		w1.backward_update_stem()

		# update global information
		self.roots.remove(word_2)
		self.rules_c.inc(rule.rule)
#		self.rules_freq.inc(rule.rule, w2.sigma)

		# update rule frequencies on the path
#		n = w1
#		while n.prev is not None:
#			for r, n2 in n.prev.next.iteritems():
#				if n2 == n:
#					self.rules_freq.inc(r, w2.sigma)
#					break
#			n = n.prev
	
	def remove_edge(self, word_1, word_2):
		self.roots.add(word_2)
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		rule = None
		for r, w in w1.next.iteritems():
			if w.word == word_2:
				rule = r
				break
		if rule is not None:
			self.rules_c[rule] -= 1
			del w1.next[rule]
		w2.prev = None

	# remove all edges
	def reset(self):
		self.rules_c = Counter()
		self.rules_freq = Counter()
		for n in self.nodes.values():
#			n.stem = n.word
			n.prev = None
			n.next = {}
			n.sigma = n.freq
			if settings.USE_WORD_FREQ:
				n.corpus_prob = float(n.freq) / self.total
			n.sum_weights = 1.0
			self.roots.add(n.word)
	
	def save_to_file(self, filename):
		def write_subtree(fp, word_1, word_2, rule):
			w2 = self.nodes[word_2]
#			write_line(fp, (word_1, word_2, rule, w2.stem, w2.freq, w2.sigma, w2.ngram_prob, w2.corpus_prob, w2.sum_weights))
			write_line(fp, (word_1, word_2, rule, w2.freq, w2.sigma, w2.ngram_prob, w2.corpus_prob, w2.sum_weights))
			for next_rule, next_word in w2.next.iteritems():
				write_subtree(fp, word_2, next_word.word, next_rule)
		with open_to_write(filename) as fp:
			for rt in self.roots:
				write_subtree(fp, u'', rt, u'')
	
	def analyses_morphochal(self, filename):
		with open_to_write(filename) as fp:
			for w in sorted(self.values(), key=lambda x: x.word):
				write_line(fp, (w.word, ' '.join(w.analysis_morphochal())))

	def save_splits(self, filename):
		for r in self.roots:
			self.nodes[r].annotate_word_structure()
		with open_to_write(filename) as fp:
			for w in sorted(self.values(), key=lambda x: x.word):
				write_line(fp, (w.word, ' '.join(w.split())))

	@staticmethod
	def init_from_file(filename):
		lexicon = Lexicon()
		unigrams = algorithms.ngrams.NGramModel(1)
		unigrams.train_from_file(filename)
		if settings.USE_WORD_FREQ:
			for word, freq in read_tsv_file(filename, (unicode, int)):
				lexicon[word] = LexiconNode(word, freq, freq, unigrams.word_prob(word), 0.0, 1.0)
				lexicon.roots.add(word)
				lexicon.total += freq
			# compute corpus probabilities
			for word in lexicon.values():
				word.corpus_prob = float(word.freq) / lexicon.total
		else:
			for word in read_tsv_file(filename, (unicode)):
				lexicon[word] = LexiconNode(word, 0, 0, unigrams.word_prob(word), 0.0, 1.0)
				lexicon.roots.add(word)
		return lexicon
	
	@staticmethod
	def load_from_file(filename):
		lexicon = Lexicon()
		for word_1, word_2, rule, freq, sigma, ngram_prob, corpus_prob, sum_weights in read_tsv_file(\
				filename, types=(unicode, unicode, unicode, int, int, float, float, float)):
			lexicon[word_2] = LexiconNode(word_2, freq, sigma, ngram_prob, corpus_prob, sum_weights)
			lexicon.total += freq
			if word_1 and rule:
				lexicon[word_2].prev = lexicon[word_1]
				lexicon[word_1].next[rule] = lexicon[word_2]
				lexicon.rules_c.inc(rule)
				lexicon.rules_freq.inc(rule, sigma)
			else:
				lexicon.roots.add(word_2)
		return lexicon
	
