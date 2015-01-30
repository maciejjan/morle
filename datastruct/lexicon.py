from datastruct.counter import *
from datastruct.rules import *
from algorithms.align import lcs
import algorithms.ngrams
from utils.files import *
import settings
import math

def freqcl(freq, maxfreq):
	try:
		return -int(math.log(float(freq) / maxfreq) / math.log(2))
	except Exception:
		print(freq, maxfreq)

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
	def __init__(self, word, freq, ngram_prob):
		self.word = word
		self.freq = freq
		self.freqcl = None
		self.ngram_prob = ngram_prob
		self.prev = None
		self.next = {}
		self.training = True
#		self.structure = None

	def root(self):
		root = self
		while root.prev is not None:
			root = root.prev
		return root
	
	def depth(self):
		if self.prev is None:
			return 0
		else:
			return 1 + self.prev.depth()
	
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
			rule = [r for r, n in node.prev.next.items() if n == node][0]
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
			rule = [r for r, n in node.prev.next.items() if n == node][0]
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
		print(space + self.word.encode('utf-8'), self.freq) #, self.sigma
		for w in self.next.values():
			w.show_tree(space=space+'\t')

	def show_tree_with_prod(self, rules, space='', prod=None, rule=None):
		print(space + (('%0.4f ' % prod) if prod else '') + self.word.encode('utf-8'), self.freq, (rule if rule else '')) #, self.sigma)
		for r, w in self.next.items():
			w.show_tree_with_prod(rules, space=space+'- ', prod=rules[r].prod, rule=r)
	
	def words_in_tree(self):
		result = [self.word]
		for w in self.next.values():
			result.extend(w.words_in_tree())
		return result
	
	def annotate_word_structure(self, depth=0):
		self.structure = [depth] * len(self.word)
		node = self
		node_depth = depth
		if self.word == u'adresowy':
			print(depth)
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
#		for i in range(len(self.structure)-1, 0, -1):
#			if self.structure[i-1] < self.structure[i] and not 0 in self.structure[:i]:
#				self.structure[i-1] = self.structure[i]
		# fix suffixes
#		for i in range(len(self.structure)-1):
#			if self.structure[i+1] < self.structure[i] and not 0 in self.structure[i:]:
#				self.structure[i+1] = self.structure[i]
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
		print(space + '|'.join(self.split()).encode('utf-8'), self.freq) #, self.sigma)
		for w in self.next.values():
			w.show_split_tree(space=space+'\t')

class Lexicon:
	def __init__(self):
		self.nodes = {}
		self.roots = set([])
		self.rules_c = Counter()
		self.max_freq = 0
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
	
	def __delitem__(self, key):
		del self.nodes[key]
	
	# TODO adding words: only with add_word!
	def add_word(self, word, freq, ngram_prob):
		self.nodes[word] = LexiconNode(word, freq, ngram_prob)
		self.total += freq
		self.roots.add(word)
		# frequency class
		if freq > self.max_freq:
			self.max_freq = freq
			self.recalculate_freqcl()
		else:
			self.nodes[word].freqcl = freqcl(freq, self.max_freq)
	
	def lexicon_logl(self, rules):
		logl = 0.0
		# lexicon log-likelihood given rules
		logl += sum([math.log(i) for i in range(1, len(self.roots)+1)])
		for root_w in self.roots:
			logl += math.log(self.nodes[root_w].ngram_prob)
		for r, count in self.rules_c.items():
			m = rules[r].domsize - count
			logl += count * math.log(rules[r].prod)
			if m > 0:
				logl += m * math.log(1.0 - rules[r].prod)
		return logl
	
#	def corpus_logl_old(self, rules):
#		logl = 0.0
#		# corpus logl-likelihood given lexicon and rules
#		for root_w in self.roots:
#			root = self.nodes[root_w]
#			logl += root.sigma * (math.log(root.sigma) - math.log(self.total))
#		for word in self.values():
#			logl -= word.sigma * math.log(word.sum_weights)
#		for rule, freq in self.rules_freq.iteritems():
#			logl += freq * math.log(rules[rule].weight)
#		logl += self.total * math.log(rules[u'#'].weight)
#		return logl
	
	# TODO rename: frequencies <- lexicon_logl
	def corpus_logl(self, rules):
		logl = 0.0
		for w1 in self.values():
			for rule, w2 in w1.next.items():
				logl += math.log(rules[rule].freqprob(w2.freqcl - w1.freqcl))
		return logl

	def logl(self, rules):
		if settings.USE_WORD_FREQ:
			return self.lexicon_logl(rules) + self.corpus_logl(rules)
		else:
			return self.lexicon_logl(rules)
	
#	def logl_gradient(self, rules):
#		d = {}
#		for r in self.rules_freq.keys():
#			d[r] = float(self.rules_freq[r]) / rules[r].weight
#		d[u'#'] = float(self.total) / rules[u'#'].weight
#		for word in self.values():
#			for r in word.next.keys():
#				d[r] -= float(word.sigma) / word.sum_weights
#			d[u'#'] -= float(word.sigma) / word.sum_weights
#		return d
	
#	def try_weights(self, rules_w):
#		logl = 0.0
#		for root_w in self.roots:
#			root = self.nodes[root_w]
#			logl += root.sigma * (math.log(root.sigma) - math.log(self.total))
#		for word in self.values():
#			logl -= word.sigma * math.log(rules_w[u'#'] +\
#				sum([rules_w[r] for r in word.next.keys()]))
#		for rule, freq in self.rules_freq.iteritems():
#			logl += freq * math.log(rules_w[rule])
#		logl += self.total * math.log(rules_w[u'#'])
#		return logl
	
	def try_edge_pr(self, word_1, word_2, rule):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		root = w1.root()
		print(word_1.encode('utf-8'), w1.sigma, w1.corpus_prob, w1.sum_weights)
		print(word_2.encode('utf-8'), w2.sigma, w2.corpus_prob, w2.sum_weights)
		print(root.word.encode('utf-8'), root.sigma, root.corpus_prob, root.sum_weights)
		print(rule.weight)
		# change of corpus log-likelihood
		result = root.sigma * (math.log(root.sigma + w2.sigma) - math.log(root.sigma))
		print(result)
		result += w2.sigma * (math.log(root.sigma + w2.sigma) - math.log(w2.sigma))
		print(result)
		result += w2.sigma * (math.log(w1.corpus_prob * w1.sum_weights * rule.weight) -\
			math.log(root.corpus_prob * root.sum_weights * (w1.sum_weights + rule.weight)))
		print(result)
		result += w1.sigma * (math.log(w1.sum_weights) - math.log(w1.sum_weights + rule.weight))
		print(result)
		# change of lexicon log-likelihood
		result += math.log(rule.prod) - math.log(len(self.roots) * w2.ngram_prob * (1.0 - rule.prod))
		print(result)
		print('')
		return result
	
	def try_edge(self, word_1, word_2, rule):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		root = w1.root()
		# change of lexicon log-likelihood
		result = math.log(rule.prod) - math.log(len(self.roots) * w2.ngram_prob * (1.0 - rule.prod))
		if settings.USE_WORD_FREQ:
			result += math.log(rule.freqprob(w2.freqcl - w1.freqcl))
		return result
	
	def draw_edge(self, word_1, word_2, rule):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		if w1.prev is not None and w1.prev.word == word_2:
			print(word_1, word_2)
			raise Exception('Cycle detected: %s, %s' % (word_1, word_2))
		root = w1.root()

		# draw the edge
		w2.prev = w1
		w1.next[rule.rule] = w2

		# update global information
		self.roots.remove(word_2)
		self.rules_c.inc(rule.rule)

	def remove_edge(self, word_1, word_2):
		self.roots.add(word_2)
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		rule = None
		for r, w in w1.next.items():
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
#		self.rules_freq = Counter()
		for n in self.nodes.values():
#			n.stem = n.word
			n.prev = None
			n.next = {}
			self.roots.add(n.word)
	
	def recalculate_freqcl(self):
		for w in self.values():
			w.freqcl = freqcl(w.freq, self.max_freq)
	
	def save_to_file(self, filename):
		def write_subtree(fp, word_1, word_2, rule):
			w2 = self.nodes[word_2]
			write_line(fp, (word_1, word_2, rule, w2.freq, w2.ngram_prob))
			for next_rule, next_word in w2.next.items():
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
	def init_from_file(filename, unigrams=None):
		lexicon = Lexicon()
		if unigrams is None:
			unigrams = algorithms.ngrams.NGramModel(1)
			unigrams.train_from_file(filename)
		if settings.USE_WORD_FREQ:
			for word, freq in read_tsv_file(filename, (str, int)):
				lexicon[word] = LexiconNode(word, freq, unigrams.word_prob(word))
				lexicon.roots.add(word)
				lexicon.total += freq
				if freq > lexicon.max_freq:
					lexicon.max_freq = freq
		else:
			for word in read_tsv_file(filename, (str, )):
				lexicon[word] = LexiconNode(word, 0, unigrams.word_prob(word))
				lexicon.roots.add(word)
		if settings.USE_WORD_FREQ:
			lexicon.recalculate_freqcl()
		return lexicon
	
	@staticmethod
	def load_from_file(filename):
		lexicon = Lexicon()
		for word_1, word_2, rule, freq, ngram_prob, in read_tsv_file(\
				filename, types=(str, str, str, int, float)):
			lexicon[word_2] = LexiconNode(word_2, freq, ngram_prob)
			lexicon.total += freq
			if freq > lexicon.max_freq:
				lexicon.max_freq = freq
			if word_1 and rule:
				lexicon[word_2].prev = lexicon[word_1]
				lexicon[word_1].next[rule] = lexicon[word_2]
				lexicon.rules_c.inc(rule)
			else:
				lexicon.roots.add(word_2)
		lexicon.recalculate_freqcl()
		return lexicon
	
