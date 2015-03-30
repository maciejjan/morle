from datastruct.counter import *
from datastruct.rules import *
from algorithms.align import lcs
import algorithms.ngrams
from utils.files import *
import settings
import copy
import heapq
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
	def __init__(self, word, freq, root_prob):
		self.word = word
		self.tag = None
		if settings.USE_TAGS:
			idx = word.rfind('_')
			if idx > -1:
				self.word = word[:idx]
				self.tag = word[idx+1:]
		self.freq = freq
		self.freqcl = None
		self.root_prob = root_prob
		self.log_root_prob = math.log(root_prob)
		self.prev = None
		self.next = {}
#		self.training = True
#		self.structure = None

	def key(self):
		return self.word + ('_'+self.tag if settings.USE_TAGS else '')

	def __lt__(self, other):
		return self.key() < other.key()
	
	def root(self):
		root = self
		while root.prev is not None:
			root = root.prev
		return root
	
	def deriving_rule(self):
		if self.prev is None:
			return None
		else:
			return self.prev.edge_label(self.key())
	
	def depth(self):
		if self.prev is None:
			return 0
		else:
			return 1 + self.prev.depth()

	def edge_label(self, key):
		for r, w in self.next.items():
			if w.key() == key:
				return r
		raise Exception('%s: no edge label for %s.' % (self.key(), key))
	
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
		print(space + self.key(), self.freq) #, self.sigma
		for w in self.next.values():
			w.show_tree(space=space+'\t')

	def show_tree_with_prod(self, rules, space='', prod=None, rule=None):
		print(space + (('%0.4f ' % prod) if prod else '') + self.word.encode('utf-8'), self.freq, (rule if rule else '')) #, self.sigma)
		for r, w in self.next.items():
			w.show_tree_with_prod(rules, space=space+'- ', prod=rules[r].prod, rule=r)
	
	def words_in_tree(self):
		result = [self.key()]
		for w in self.next.values():
			result.extend(w.words_in_tree())
		return result
	
	def search(self):
		'''Depth-first search.'''
		yield self
		for n in self.next.values():
			for x in n.search():
				yield x
	
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
	def __init__(self, rootdist=None, ruleset=None):
		self.nodes = {}
		self.roots = set([])
		self.rootdist = rootdist
		self.rules_c = Counter()		# counter for rule frequencies
		if ruleset is not None:
			self.ruleset = ruleset
		else:
			self.ruleset = RuleSet()
		self.max_freq = 0				# the largest word freqency
	
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

	def get_edges_for_rule(self, rule):
		result = []
		for w in self.nodes.values():
			if rule in w.next:
				result.append((w, w.next[rule]))
		return result
	
	# TODO adding words: only with add_word!
	def add_word(self, word, freq, root_prob=None):
		if root_prob is None:
			root_prob = self.rootdist.word_prob(word)
		self.nodes[word] = LexiconNode(word, freq, root_prob)
#		self.total += freq
		self.roots.add(word)
		# frequency class
		if settings.USE_WORD_FREQ:
			if freq > self.max_freq:
				self.max_freq = freq
				self.recalculate_freqcl()
			else:
				self.nodes[word].freqcl = freqcl(freq, self.max_freq)
	
	def logl(self):
		result = 0
		# probabilities of the roots
		plambda = len(self)/2
		log_plambda = math.log(plambda)
		for root in self.roots:
			result += self.nodes[root].log_root_prob
			result += log_plambda
		result -= plambda
		# count rule prior probabilities and edge probabilities
		for rule, freq in self.rules_c.items():
			result += self.ruleset.rule_cost(rule, freq)
		# add frequency class probabilities of the resulting words
		if settings.USE_WORD_FREQ:
			for w1 in self.values():
				for rule, w2 in w1.next.items():
					result += math.log(self.ruleset[rule].freqprob(w2.freqcl - w1.freqcl))
		return result
	
	def try_edge(self, word_1, word_2, rule):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		r = self.ruleset[rule]
		plambda = len(self)/2
		# change of lexicon log-likelihood
		result = math.log(r.prod) - math.log(plambda * w2.root_prob * (1.0 - r.prod))
		if settings.USE_WORD_FREQ:
			result += math.log(r.freqprob(w2.freqcl - w1.freqcl))
		return result
	
	def draw_edge(self, word_1, word_2, rule):
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		if w1.prev is not None and w1.prev.key() == word_2:
			raise Exception('Cycle detected: %s, %s' % (word_1, word_2))
		if w2.prev is not None:
			raise Exception('draw_edge: %s has already got an ingoing edge.' % word_2)
		if rule in w1.next:
			raise Exception('draw_edge: %s has already got an outgoing edge %s: %s.' %\
				(word_1, rule, w1.next[rule].key()))
		# draw the edge
		w2.prev = w1
		w1.next[rule] = w2
		# update global information
		self.roots.remove(word_2)
		self.rules_c.inc(rule)

	def remove_edge(self, word_1, word_2):
		self.roots.add(word_2)
		w1, w2 = self.nodes[word_1], self.nodes[word_2]
		rule = w1.edge_label(word_2)
		self.rules_c[rule] -= 1
		del w1.next[rule]
		w2.prev = None
	
	def try_word(self, word, max_depth=None, max_results=1, ignore_lex_depth=False):
		'''Compute the minimum cost of adding a new word.'''
		# TODO frequency class

		def cost_with_rule(r):
			return math.log(r.prod)-math.log(1-r.prod)

		def expand_queue(queue, cost, subtree, base, r, depth):
			# add as a root
			base_prob = self.rootdist.word_prob(base)
			new_node = LexiconNode(base, 0, base_prob)
			#   link the new node into the subtree
			if subtree is not None and r is not None:
				subtree = copy.deepcopy(subtree)
				new_node.next[r.rule] = subtree
				subtree.prev = new_node
			#   fill in the missing tag with the most common one
			if settings.USE_TAGS and new_node.tag is None:
				new_node_t = copy.copy(new_node)
				new_node_t.tag = max(list(self.rootdist.tags.items()),\
				                     key=lambda x:x[1])[0]
				base_prob_t = self.rootdist.word_prob(new_node_t.key())
				new_node_t.root_prob = base_prob_t
				base_cost_t = math.log(base_prob_t)
				heapq.heappush(queue, (cost - base_cost_t,\
									   new_node_t, new_node_t.key(), None, depth))
			else:	# tag already provided
				base_cost = math.log(base_prob)
				heapq.heappush(queue, (cost - base_cost,\
									   new_node, base, None, depth))
			# add all possible derivations
			if max_depth is None or depth < max_depth:
				for r2 in self.ruleset.values():
					base_t = base + ('_' + r2.rule_obj.tag[1]\
							if settings.USE_TAGS and new_node.tag is None\
							else '')
					if r2.rule_obj.rmatch(base_t):
						if settings.USE_TAGS and new_node.tag is None:
							# fill in the missing tag
							new_node_t = copy.copy(new_node)
							new_node_t.tag = r2.rule_obj.tag[1]
							base_t = base + '_' + new_node_t.tag
							heapq.heappush(queue, (cost - cost_with_rule(r2),\
												   new_node_t, base_t, r2, depth+1))
						else:		# tag already provided
							heapq.heappush(queue, (cost - cost_with_rule(r2),\
												   new_node, base, r2, depth+1))

		results, queue, max_cost = [], [], 0.0

		# if the word is contained in lexicon, add it to results
		if word in self.nodes:
			results.append((None, None, self.nodes[word], 0.0))
		if settings.USE_TAGS and word.rfind('_') == -1:
			for word2 in self.nodes.keys():
				if word2.startswith(word+'_'):
					results.append((None, None, self.nodes[word2], 0.0))

		# initialize the queue
		expand_queue(queue, 0.0, None, word, None, 0)

		# main loop: process the queue entries
		while queue:
			n_cost, n_subtree, n_word, n_r, n_depth = heapq.heappop(queue)
			
			# if results full and next cost bigger -> break and return
			if len(results) >= max_results and n_cost > max_cost:
				break

			# if the current node is a root -> append to results
			if n_r is None:
				results.append((None, None, n_subtree, n_cost))
				max_cost = max(max_cost, n_cost)
			# else: generate possible bases and analyze them further
			else:
				for n_base in n_r.rule_obj.reverse().apply(n_word):
					if n_base in self.nodes and\
							((ignore_lex_depth and n_depth <= max_depth) or\
							(self.nodes[n_base].depth() + n_depth <= max_depth)):
						results.append((n_base, n_r.rule, n_subtree, n_cost))
						max_cost = max(max_cost, n_cost)
					else:
						expand_queue(queue, n_cost, n_subtree, n_base, n_r, n_depth)

		results.sort(key=lambda x: x[3])		# sort results according to cost
		return results[:max_results]
	
	def generate(self, base, tag):
		'''Generate an inflected form for a given base and tag.'''

		def cost_with_rule(r):
			return math.log(r.prod)-math.log(1-r.prod)

		if base in self.nodes:
			for n in self.nodes[base].next.values():
				if n.tag == tag:
					return n.key(), 0.0
		results = []
		for r in self.ruleset.values():
			if r.rule_obj.tag[1] == tag and r.rule_obj.lmatch(base):
				results.extend([(word, -cost_with_rule(r))\
					for word in r.rule_obj.apply(base)])
		return min(results, key=lambda x: x[1]) if results else None

	def expand(self, max_words=None, max_cost=0.0):
		'''Generate new words with low cost.'''

		def cost_with_rule(r):
			return math.log(r.prod)-math.log(1-r.prod)

		max_freqcl = max([w.freqcl for w in self.values()])\
			if settings.USE_WORD_FREQ else None
		trh = TrigramHash()
		for word in self.nodes.keys():
			trh.add(word)
		rules_queue = [r for r in self.ruleset.values()\
			if cost_with_rule(r) > max_cost]
		rules_queue.sort(reverse=True, key=lambda r: cost_with_rule(r))

		results = []
		for r in rules_queue:
			trigrams = r.rule_obj.get_trigrams()
			words = None
			if not trigrams:
				words = list(self.keys())
			else:
				words = trh.retrieve(trigrams[0])
				for tr in trigrams[1:]:
					words &= trh.retrieve(tr)
			for word in words:
				w = self.nodes[word]
				if r.rule not in w.next:
					cost = cost_with_rule(r)
					if settings.USE_WORD_FREQ:
						freqcl = max_freqcl + 1 - w.freqcl
						cost += r.freqprob(freqcl)
					results.extend([(word2, cost) \
						for word2 in r.rule_obj.apply(word)\
							if word2 not in self.nodes and cost > max_cost])

		if settings.USE_WORD_FREQ:
			results.sort(reverse=True, key=lambda x: x[1])
		return results

	def reset(self):
		'''Remove all edges.'''
		self.rules_c = Counter()
		for n in self.nodes.values():
			n.prev = None
			n.next = {}
			self.roots.add(n.key())
	
	def recalculate_freqcl(self):
		for w in self.values():
			w.freqcl = freqcl(w.freq, self.max_freq)
	
	def save_to_file(self, filename):
		def write_subtree(fp, word_1, word_2, rule):
			w2 = self.nodes[word_2]
			write_line(fp, (word_1, word_2, rule, w2.freq, w2.root_prob))
			for next_rule, next_word in w2.next.items():
				write_subtree(fp, word_2, next_word.key(), next_rule)
		with open_to_write(filename) as fp:
			for rt in self.roots:
				write_subtree(fp, '', rt, '')
	
	def analyses_morphochal(self, filename):
		with open_to_write(filename) as fp:
			for w in sorted(self.values(), key=lambda x: x.key()):
				write_line(fp, (w.key(), ' '.join(w.analysis_morphochal())))

	def save_splits(self, filename):
		for r in self.roots:
			self.nodes[r].annotate_word_structure()
		with open_to_write(filename) as fp:
			for w in sorted(self.values(), key=lambda x: x.word):
				write_line(fp, (w.word, ' '.join(w.split())))

	@staticmethod
	def init_from_file(filename, rootdist=None, ruleset=None):	# TODO
		'''Create a lexicon with no edges from a wordlist.'''
		if rootdist is None:
			rootdist = algorithms.ngrams.NGramModel(settings.ROOTDIST_N)	# TODO what about tags?
			rootdist.train_from_file(filename)
		lexicon = Lexicon(rootdist, ruleset)
		if settings.USE_WORD_FREQ:
			for word, freq in read_tsv_file(filename, (str, int)):
				lexicon[word] = LexiconNode(word, freq, rootdist.word_prob(word))
				lexicon.roots.add(word)
				if freq > lexicon.max_freq:
					lexicon.max_freq = freq
		else:
			for (word, ) in read_tsv_file(filename, (str, )):
				lexicon[word] = LexiconNode(word, 0, rootdist.word_prob(word))
				lexicon.roots.add(word)
		if settings.USE_WORD_FREQ:
			lexicon.recalculate_freqcl()
		return lexicon
	
	@staticmethod
	def load_from_file(filename, ruleset):
		lexicon = Lexicon(ruleset=ruleset)
		for word_1, word_2, rule, freq, root_prob, in read_tsv_file(\
				filename, types=(str, str, str, int, float)):
			lexicon[word_2] = LexiconNode(word_2, freq, root_prob)
			lexicon.roots.add(word_2)
#			lexicon.total += freq
			if freq > lexicon.max_freq:
				lexicon.max_freq = freq
			if word_1 and rule and rule in ruleset:
				lexicon.draw_edge(word_1, word_2, rule)
		if settings.USE_WORD_FREQ:
			lexicon.recalculate_freqcl()
		# train roots distribution
		lexicon.rootdist = NGramModel(settings.ROOTDIST_N)
		lexicon.rootdist.train([(lexicon[rt].key(), lexicon[rt].freq)\
			for rt in lexicon.roots])
		return lexicon

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
	
