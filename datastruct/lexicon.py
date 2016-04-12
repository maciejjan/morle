#from datastruct.rules import *
#from algorithms.align import lcs
#import algorithms.ngrams
import algorithms.fst
from utils.files import *
import settings
from collections import defaultdict
import itertools
import re
#import copy
#import heapq
#import math

#def freqcl(freq, maxfreq):
#	try:
#		return -int(math.log(float(freq) / maxfreq) / math.log(2))
#	except Exception:
#		print(freq, maxfreq)
#
#def align_words(word_1, word_2):
#	cs = algorithms.align.lcs(word_1, word_2)
#	pattern = re.compile('(.*)' + '(.*?)'.join([\
#		letter for letter in cs]) + '(.*)')
#	m1 = pattern.search(word_1)
#	m2 = pattern.search(word_2)
#	alignment = []
#	for i, (x, y) in enumerate(zip(m1.groups(), m2.groups())):
#		if x or y:
#			alignment.append((x, y))
#		if i < len(cs):
#			alignment.append((cs[i], cs[i]))
#	return alignment

def tokenize_word(string):
	'''Separate a string into a word and a POS-tag,
	   both expressed as sequences of symbols.'''
	m = re.match(settings.WORD_PATTERN_CMP, string)
	if m is None:
		raise Exception('Error while tokenizing word: %s' % string)
	return tuple(re.findall(settings.SYMBOL_PATTERN_CMP, m.group('word'))),\
		   tuple(re.findall(settings.TAG_PATTERN_CMP, m.group('tag')))

class LexiconEdge:
	def __init__(self, word_1, word_2, rule, weight=0.0):
		self.word_1 = word_1
		self.word_2 = word_2
		self.rule = rule
		self.weight = weight

class LexiconNode:
	def __init__(self, word, freq=None, vec=None):
		self.word, self.tag = tokenize_word(word)
		if settings.WORD_FREQ_WEIGHT > 0:
			self.freq = freq
		if settings.WORD_VEC_WEIGHT > 0:
			self.vec = vec
		self.parent = None
		self.edges = []
		self.transducer = None
#		self.training = True
#		self.structure = None

	def key(self):
		return ''.join(self.word + self.tag)
#		result = self.word
#		if settings.USE_TAGS and self.tag is not None:
#			result += ''.join(self.tag)
#		return result
	
	def __lt__(self, other):
		return self.key() < other.key()
	
	def root(self):
		root = self
		while root.prev is not None:
			root = root.prev
		return root
	
	def has_ancestor(self, word):
		if self.prev is None:
			return False
		else:
			if self.prev.word == word:
				return True
			else:
				return self.prev.has_ancestor(word)
	
	def deriving_rule(self):
		if self.prev is None:
			return None
		else:
			return self.prev.edge_label(self.key())
	
	def build_transducer(self, alphabet=None):
		seq = self.word + self.tag
		self.transducer =\
			algorithms.fst.seq_to_transducer(zip(seq, seq), alphabet=alphabet)
	
#	def depth(self):
#		if self.prev is None:
#			return 0
#		else:
#			return 1 + self.prev.depth()

#	def edge_label(self, key):
#		for r, w in self.next.items():
#			if w.key() == key:
#				return r
#		raise Exception('%s: no edge label for %s.' % (self.key(), key))
	
	def analysis(self):
		analysis = []
		node = self
		while node.prev is not None:
			node = node.prev
			analysis.append(node.key())
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
	
#	def analysis_morphochal(self):
#		analysis = []
#		node = self
#		while node.prev is not None:
#			word = node.prev.word
#			rule = [r for r, n in node.prev.next.items() if n == node][0]
#			rule_sp = rule.split('*')
#			if len(rule_sp) == 3:
#				rule = rule_sp[0] + '*' + rule_sp[2]
##				rule = rule_sp[0] + rule_sp[2]
#				analysis.append(rule_sp[1])
#			else:
#				analysis.append(rule)
##			rule = Rule.from_string(rule)
##			if rule.prefix[0]:
##				analysis.append(rule.prefix[0] + '-')
##			if rule.prefix[1]:
##				analysis.append(rule.prefix[1] + '+')
##			if rule.suffix[0]:
##				analysis.append('-' + rule.suffix[0])
##			if rule.suffix[1]:
##				analysis.append('+' + rule.suffix[1])
##			for x, y in rule.alternations:
##				if x:
##					analysis.append('-' + x + '-')
##				if y:
##					analysis.append('+' + y + '+')
#			node = node.prev
#		analysis.append(node.word)
#		to_remove = []
#		for x in analysis:
#			if x.find('-') > -1 and x.replace('-', '+') in analysis:
#				to_remove.append(x)
#				to_remove.append(x.replace('-', '+'))
#			elif x.find('-') > -1 and x.replace('-', '') in analysis:
#				to_remove.append(x)
#				to_remove.append(x.replace('-', ''))
#		for x in to_remove:
#			if x in analysis:
#				analysis.remove(x)
#		analysis = [x for x in analysis \
#			if (not '+' in x and not '-' in x)\
#				or (x.replace('+', '') in self.word)]
#		analysis.reverse()
#		return analysis

	def show_tree(self, space=''):
		print(space + self.key(), self.freq) #, self.sigma
		for w in self.next.values():
			w.show_tree(space=space+'\t')

#	def show_tree_with_prod(self, rules, space='', prod=None, rule=None):
#		print(space + (('%0.4f ' % prod) if prod else '') + self.word.encode('utf-8'), self.freq, (rule if rule else '')) #, self.sigma)
#		for r, w in self.next.items():
#			w.show_tree_with_prod(rules, space=space+'- ', prod=rules[r].prod, rule=r)
	
#	def words_in_tree(self):
#		result = [self.key()]
#		for w in self.next.values():
#			result.extend(w.words_in_tree())
#		return result
	
#	def search(self):
#		'''Depth-first search.'''
#		yield self
#		for n in self.next.values():
#			for x in n.search():
#				yield x
	
#	def annotate_word_structure(self, depth=0):
#		self.structure = [depth] * len(self.word)
#		node = self
#		node_depth = depth
#		if self.word == u'adresowy':
#			print(depth)
#		while node.prev is not None:
#			node = node.prev
#			node_depth -= 1
#			alignment = align_words(self.word, node.word)
#			i = 0
#			for x, y in alignment:
#				if x == y:
#					self.structure[i] = node_depth
#					i += 1
#				else:
#					i += len(x)
		# fix prefixes
#		for i in range(len(self.structure)-1, 0, -1):
#			if self.structure[i-1] < self.structure[i] and not 0 in self.structure[:i]:
#				self.structure[i-1] = self.structure[i]
		# fix suffixes
#		for i in range(len(self.structure)-1):
#			if self.structure[i+1] < self.structure[i] and not 0 in self.structure[i:]:
#				self.structure[i+1] = self.structure[i]
#		for child in self.next.values():
#			child.annotate_word_structure(depth+1)
	
#	def split(self):
#		split = []
#		cur_morph = self.word[0]
#		for i in range(1, len(self.structure)):
#			if self.structure[i] == self.structure[i-1]:
#				cur_morph += self.word[i]
#			else:
#				split.append(cur_morph)
#				cur_morph = self.word[i]
#		split.append(cur_morph)
#		return split
#
#	def show_split_tree(self, space=''):
#		print(space + '|'.join(self.split()).encode('utf-8'), self.freq) #, self.sigma)
#		for w in self.next.values():
#			w.show_split_tree(space=space+'\t')

#TODO priors on vector representation etc.
class Lexicon:
	def __init__(self, rootdist=None, ruleset=None):
		self.nodes = {}
		self.roots = set()
#		self.roots = set([])
#		self.rootdist = rootdist
		self.rules_c = defaultdict(lambda: 0)
		self.transducer = None		# TODO
#		if ruleset is not None:
#			self.ruleset = ruleset
#		else:
#			self.ruleset = RuleSet()
#		self.max_freq = 0				# the largest word freqency
	
	def __len__(self):
		return len(self.nodes)
	
	def __contains__(self, key):
		return self.nodes.__contains__(key)
	
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

#	def get_edges_for_rule(self, rule):
#		result = []
#		for w in self.nodes.values():
#			if rule in w.next:
#				result.append((w, w.next[rule]))
#		return result
	
	# TODO adding words: only with add_node!
	def add_node(self, node):
		key = node.key()
		self.nodes[key] = node
		self.roots.add(key)
#		if root_prob is None:
#			root_prob = self.rootdist.word_prob(word)
#		self.nodes[word] = LexiconNode(word, freq, root_prob)
#		self.total += freq
#		self.roots.add(word)
		# frequency class
#		if settings.USE_WORD_FREQ:
#			if freq > self.max_freq:
#				self.max_freq = freq
#				self.recalculate_freqcl()
#			else:
#				self.nodes[word].freqcl = freqcl(freq, self.max_freq)
	
	# TODO store logl, don't calculate it each time!
#	def logl(self):
#		result = 0
#		# probabilities of the roots
#		plambda = len(self)/2
#		log_plambda = math.log(plambda)
#		for root in self.roots:
#			result += self.nodes[root].log_root_prob
#			result += log_plambda
#		result -= plambda
#		# count rule prior probabilities and edge probabilities
#		for rule, freq in self.rules_c.items():
#			result += self.ruleset.rule_cost(rule, freq)
#		# add frequency class probabilities of the resulting words
#		if settings.USE_WORD_FREQ:
#			for w1 in self.values():
#				for rule, w2 in w1.next.items():
#					result += math.log(self.ruleset[rule].freqprob(w2.freqcl - w1.freqcl))
#		if settings.WORD_VEC_FACTOR > 0:
#			pass	# TODO word2vec
#		return result
	
#	def try_edge(self, word_1, word_2, rule):
#		w1, w2 = self.nodes[word_1], self.nodes[word_2]
#		r = self.ruleset[rule]
#		plambda = len(self)/2
#		# change of lexicon log-likelihood
#		result = math.log(r.prod) - math.log(plambda * w2.root_prob * (1.0 - r.prod))
#		if settings.USE_WORD_FREQ:
#			result += math.log(r.freqprob(w2.freqcl - w1.freqcl))
#		if settings.WORD_VEC_FACTOR > 0:
#			result += math.log(r.vecprob())
#			pass	# TODO consider word vector
#		return result
	
#	def edge_possible(self, word_1, word_2, rule):
#		w1, w2 = self.nodes[word_1], self.nodes[word_2]
#		if w1.prev is not None and w1.prev.key() == word_2:
#			return False
#		if w2.prev is not None:
#			return False
#		if rule in w1.next:
#			return False
#		# TODO use has_ancestor()?
#		return True

	# TODO full cycle detection
	def check_if_edge_possible(self, edge):
		w1, w2 = self.nodes[edge.word_1], self.nodes[edge.word_2]
		if w1.parent is not None and w1.parent.key() == edge.word_2:
			raise Exception('Cycle detected: %s, %s' % (edge.word_1, edge.word_2))
		if w2.parent is not None:
			raise Exception('draw_edge: %s has already got an ingoing edge.' % edge.word_2)

	def add_edge(self, edge):
		self.check_if_edge_possible(edge)
		w1, w2 = self.nodes[edge.word_1], self.nodes[edge.word_2]
		w1.edges.append(edge)
		w2.parent = w1
		self.roots.remove(edge.word_2)
		self.rules_c[edge.rule] += 1
	
#	def draw_edge(self, word_1, word_2, rule):
#		w1, w2 = self.nodes[word_1], self.nodes[word_2]
#		if w1.prev is not None and w1.prev.key() == word_2:
#			raise Exception('Cycle detected: %s, %s' % (word_1, word_2))
#		if w2.prev is not None:
#			raise Exception('draw_edge: %s has already got an ingoing edge.' % word_2)
#		if rule in w1.next:
#			raise Exception('draw_edge: %s has already got an outgoing edge %s: %s.' %\
#				(word_1, rule, w1.next[rule].key()))
#		# draw the edge
#		w2.prev = w1
#		w1.next[rule] = w2
#		# update global information
#		self.roots.remove(word_2)
#		self.rules_c.inc(rule)

#	def remove_edge(self, word_1, word_2):
#		self.roots.add(word_2)
#		w1, w2 = self.nodes[word_1], self.nodes[word_2]
#		rule = w1.edge_label(word_2)
#		self.rules_c[rule] -= 1
#		del w1.next[rule]
#		w2.prev = None
	
#	def reset(self):
#		'''Remove all edges.'''
#		self.rules_c = Counter()
#		for n in self.nodes.values():
#			n.prev = None
#			n.next = {}
#			self.roots.add(n.key())
#	
#	def recalculate_freqcl(self):
#		for w in self.values():
#			w.freqcl = freqcl(w.freq, self.max_freq)

	def build_transducer(self):
		alphabet =\
			tuple(sorted(set(
				itertools.chain(*(n.word+n.tag for n in self.nodes.values()))
			)))
		print(alphabet)
		for n in self.nodes.values():
			n.build_transducer(alphabet=alphabet)
		self.transducer =\
			algorithms.fst.binary_disjunct(
				n.transducer for n in self.nodes.values()
			)
	
	def save_to_file(self, filename):
		def write_subtree(fp, word_1, word_2, rule):
			w2 = self.nodes[word_2]
			write_line(fp, (word_1, word_2, rule, w2.freq,\
			                settings.VECTOR_SEP.join(map(str, w2.vec))))
			for edge in w2.edges:
				write_subtree(fp, word_2, edge.word_2, edge.rule)
		with open_to_write(filename) as fp:
			for rt in self.roots:
				write_subtree(fp, '', rt, '')
	
	@staticmethod
	def init_from_file(filename):
		'''Create a lexicon with no edges from a wordlist.'''
		# TODO rootdist

#		if rootdist is None:
#			rootdist = algorithms.ngrams.NGramModel(settings.ROOTDIST_N)
#			rootdist.train_from_file(filename)
		lexicon = Lexicon()
		# read the input file and create nodes
		for node_data in read_tsv_file(filename, settings.WORDLIST_FORMAT):
			lexicon.add_node(LexiconNode(*node_data))
		return lexicon

	@staticmethod
	def load_from_file(filename):
		lexicon = Lexicon()
		for line in read_tsv_file(filename, settings.LEXICON_FORMAT):
			node_data = (line[1],) + line[3:]
			lexicon.add_node(LexiconNode(*node_data))
			if line[0] and line[2]:
				lexicon.add_edge(LexiconEdge(*line[:3]))
		return lexicon
		# TODO rootdist

	def save_model(self, rules_file, lexicon_file):
		'''Save rules and lexicon.'''
		self.ruleset.save_to_file(rules_file)
		self.save_to_file(lexicon_file)

	@staticmethod
	def load_model(rules_file, lexicon_file):
		'''Load rules and lexicon.'''
		ruleset = RuleSet.load_from_file(rules_file)
		lexicon = Lexicon.load_from_file(lexicon_file, ruleset)
		return lexicon
	
