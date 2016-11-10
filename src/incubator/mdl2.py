from datastruct.counter import *
from utils.files import *
from utils.printer import *
import incubator.ngrams as ngr
import math
import pickle
import settings

NUM_ITERATIONS = 500
GAMMA_THRESHOLD = 1e-15
GRAPH_MDL_FILE = 'graph_mdl.txt'
ANALYSES_FILE = 'analyses.txt'
RULES_FILE = 'rules.txt'
LEXICON_FILE = 'lexicon.txt'
LEXICON_P_FILE = 'lexicon_p.txt'

class Rule:
	# fields:
	# tr - transformation
	# prod - productivity
	# weight
	# domsize - number of words, for which the rule could apply
	# phi - number of times the rules is applied
	def __init__(self, tr, prod, weight, domsize):
		self.tr = tr
		self.prod = prod
		self.weight = weight
		self.domsize = domsize
		self.phi = 0

class LexiconNode:
	# fields:
	# word
	# sigma - sum of frequencies in the subtree
	# sum_weights - sum of rule weights of outgoing edges plus the weight of #
	# prev - previous node
	# next - dictionary rule:next node
	def __init__(self, word, ngr_prob, prob, freq, sigma, sum_weights):
		self.word = word
		self.ngr_prob = ngr_prob
		self.prob = prob
		self.freq = freq
		self.sigma = sigma
		self.sum_weights = sum_weights
		self.prev = None
		self.next = {}
	
	def stem(self):
		stem = self
		while stem.prev is not None:
			stem = stem.prev
		return stem
	
	def analysis(self):
		analysis = []
		stem = self
		while stem.prev is not None:
			stem = stem.prev
			analysis.append(stem.word)
		return analysis
	
	def forward_multiply_prob(self, p):
		self.prob *= p
		for child in self.next.values():
			child.forward_multiply_prob(p)
	
	def backward_add_sigma(self, s):
		self.sigma += s
		if self.prev is not None:
			self.prev.backward_add_sigma(s)

def get_edge_score(lexicon, rules, word_1, word_2, rule):
	r = rules[rule]
	w1, w2 = lexicon[word_1], lexicon[word_2]
	s = w1.stem()
#	print s.word.encode('utf-8'), s.sigma
#	print word_1.encode('utf-8'), '->', word_2.encode('utf-8')
	log_cpr = s.sigma * (math.log(s.sigma + w2.sigma) - math.log(s.sigma))
#	print log_cpr
	log_cpr += w2.sigma * (math.log(s.sigma + w2.sigma) - math.log(w2.sigma))
#	print log_cpr
	log_cpr += w2.sigma * (math.log(w1.prob * w1.sum_weights * r.weight) -\
		math.log(s.prob * s.sum_weights * (w1.sum_weights + r.weight)))
#	print log_cpr
	log_cpr += w1.sigma * (math.log(w1.sum_weights) - math.log(w1.sum_weights + r.weight))
#	print log_cpr
#	print math.log(r.prod / (w2.ngr_prob * (1.0 - r.prod))), log_cpr
	log_n_max = math.log(r.prod / (w2.ngr_prob * (1.0 - r.prod))) + log_cpr
#	if log_n_max > 0:
#		print '+', '('+s.word.encode('utf-8')+ ' -> )',\
#			word_1.encode('utf-8'), '->', word_2.encode('utf-8'), log_n_max
#	else:
#		print log_n_max
	return log_n_max

def draw_edge(lexicon, rules, word_1, word_2, rule):
	w1, w2 = lexicon[word_1], lexicon[word_2]
	s = w1.stem()
	r = rules[rule]

	# update word probabilities
	w2.forward_multiply_prob(w1.prob * w1.sum_weights * r.weight /\
		(s.prob * s.sum_weights * (w1.sum_weights + r.weight)))
	s.forward_multiply_prob(float(s.sigma + w2.sigma) / s.sigma)
	w2.forward_multiply_prob(float(s.sigma + w2.sigma) / w2.sigma)
	w1.forward_multiply_prob(float(w1.sum_weights) / (w1.sum_weights + r.weight))

	# update frequency and weight sums
	w1.backward_add_sigma(w2.sigma)
	w1.sum_weights += r.weight

	w2.prev = w1
	w1.next[rule] = w2

def build_lexicon(rules, corpus, unigrams):
	# init lexicon
	lexicon = {}
	sum_freq = sum(corpus.values())
	for word, freq in corpus.iteritems():
		lexicon[word] = LexiconNode(word,\
			ngr.word_prob(word, unigrams, 1),\
			float(freq) / sum_freq,\
			freq, freq, rules[u'#'].weight)
	# compute best n_max for each word and store it in the memory
	word_n_max = {}
	for word_1, word_2, rule in load_tsv_file(settings.FILES['surface.graph'],\
			print_progress=False):
#			print_progress=True, print_msg='Computing edge scores...'):
		if rules.has_key(rule):
			n_max = get_edge_score(lexicon, rules, word_1, word_2, rule)
			if n_max < 0:
				continue
			if not word_n_max.has_key(word_2) or word_n_max[word_2] < n_max:
				word_n_max[word_2] = n_max
#	# TODO
#	for word, n_max in sorted(word_n_max.iteritems(), reverse=True, key=lambda x:x[1]):
#		print word, n_max
	# sort the words according to n_max and remember the position of each word
	word_pos = dict([(word, pos) for pos, (word, n_max) in enumerate(\
		sorted(word_n_max.iteritems(), reverse=True, key=lambda x:x[1]), 1)])
	# write the desired position for each word into the graph
	with open_to_write(GRAPH_MDL_FILE) as outfp:
		for word_1, word_2, rule in load_tsv_file(settings.FILES['surface.graph'],\
				print_progress=False):
#				print_progress=True, print_msg='Writing edge scores...'):
			if word_pos.has_key(word_2) and rules.has_key(rule):
				write_line(outfp, (word_1, word_2, rule, word_pos[word_2]))
	# sort the graph according to desired positions of words
	sort_file(GRAPH_MDL_FILE, numeric=True, key=4)
	# for each word, compute once more the n_max for each possible edge and select the best one
	n = len(lexicon)
	with open_to_write('lexicon.txt.r') as outfp:
		for word_2, rest in load_tsv_file_by_key(GRAPH_MDL_FILE, 2,\
				print_progress=False):
#				print_progress=True, print_msg='Building lexicon...'):
			edges = sorted([(word_1, word_2, rule,\
				get_edge_score(lexicon, rules, word_1, word_2, rule))\
				for (word_1, rule, _) in rest],\
				reverse=True, key = lambda x: x[3])
			for edge in edges:
				if math.log(n) < edge[3] and not edge[1] in lexicon[edge[0]].analysis():
#					print edge
					draw_edge(lexicon, rules, edge[0], edge[1], edge[2])
					n -= 1
					write_line(outfp, (edge[1], edge[0], edge[2], edge[3]))
					break
	return lexicon

def reestimate_rule_prod(rules, lexicon):
	# count occurrences of rules in the lexicon
	rules_c = Counter()
	for word in lexicon.values():
		for r in word.next.keys():
			rules_c.inc(r)
	# replace productivity with count / domsize
	for r in rules.keys():
		if rules_c.has_key(r):
			rules[r].prod = float(rules_c[r]) / rules[r].domsize
		elif r == u'#':
			pass
		else:
			del rules[r]

def count_rules_freq(lexicon):
	rules_f = Counter()
	total = 0
	for word in lexicon.values():
		for r, w in word.next.iteritems():
			rules_f.inc(r, count=w.sigma)
		if word.prev is None:
			total += word.sigma
	rules_f[u'#'] = total
	return rules_f

# log-likelihood of the lexicon
def lex_logl(rules, lexicon):
	# count occurrences of rules in the lexicon
	rules_c = Counter()
	for word in lexicon.values():
		for r in word.next.keys():
			rules_c.inc(r)
	# stem probabilities
	logl = 0.0
	n = 1
	for word in lexicon.values():
		if word.prev is None:
			logl += math.log(n) + math.log(word.ngr_prob)
			n += 1
	print 'Lex-stems:', logl
	# rule productivities
	for r, count in rules_c.iteritems():
		m = rules[r].domsize - count
		logl += count * math.log(rules[r].prod) +\
			m * math.log(1.0 - rules[r].prod)
	return logl

# calculate the log-likelihood of the corpus given rules and lexicon
# (constant terms are omitted)
def logl(rules_w, rules_f, lexicon):
	logl = 0.0
	total = sum([word.sigma for word in lexicon.values() if word.prev is None])
	for word in lexicon.values():
		if word.prev is None:
			logl += word.sigma * (math.log(word.sigma) - math.log(total))
#		logl -= word.sigma * math.log(word.sum_weights)
		logl -= word.sigma * math.log(rules_w[u'#'] +\
			sum([rules_w[r] for r in word.next.keys()]))
	for rule, freq in rules_f.iteritems():
		logl += freq * math.log(rules_w[rule])
	logl += total * math.log(rules_w[u'#'])
	return logl

# calculate the vector of partial derivatives of the log-likelihood
def delta_logl(rules_w, rules_f, lexicon):
	d = {}
	for rule in rules_w.keys():
		d[rule] = float(rules_f[rule]) / rules_w[rule]
	for word in lexicon.values():
		for rule in word.next.keys():
			d[rule] -= float(word.sigma) / word.sum_weights
		d[u'#'] -= float(word.freq) / word.sum_weights
	return d

def reestimate_rule_weights(rules, lexicon):
	# gradient descent
	rules_w = dict([(r.tr, r.weight) for r in rules.values()])
	rules_f = count_rules_freq(lexicon)
	print sum(rules_f.values()), sum([w.sigma for w in lexicon.values()])
	old_logl = logl(rules_w, rules_f, lexicon)
	while True:
		d = delta_logl(rules_w, rules_f, lexicon)
		new_rules_w = {}
		gamma = 1000.0
		while gamma >= GAMMA_THRESHOLD:
#			print 'gamma = ', gamma
			gamma_too_big = False
			for r in rules_w.keys():
				new_rules_w[r] = rules_w[r] + gamma * d[r]
				if new_rules_w[r] <= 0.0:
					gamma_too_big = True
#					print r, d[r]
					break
			if gamma_too_big:
				gamma /= 10
				continue
#			else:
#				print 'ok'
			new_logl = logl(new_rules_w, rules_f, lexicon)
			if new_logl > old_logl + 100.0:
				print 'improvement', new_logl, 'over', old_logl, 'with gamma =', gamma
				rules_w = new_rules_w
				old_logl = new_logl
				break
			else:
#				print 'no improvement', new_logl, old_logl
				gamma /= 10
		if gamma < GAMMA_THRESHOLD:
			break
	lexicon_logl = lex_logl(rules, lexicon)
	print 'LogL:', lexicon_logl + old_logl, '( lex:', lexicon_logl,\
		', corp:', old_logl, ')'
	# normalize the rule weight
	print rules_w[u'#']
	print logl(rules_w, rules_f, lexicon)
	for r in rules.values():
		r.weight = rules_w[r.tr] / rules_w[u'#']
		if r.tr != u'#':
			rules_w[r.tr] = r.weight
	rules_w[u'#'] = 1.0
	print logl(rules_w, rules_f, lexicon)

def load_rules(filename):
	# rules file: rule, prod, weight, domsize
	rules = {}
	for rule, prod, weight, domsize in load_tsv_file(filename):
		rules[rule] = Rule(rule, float(prod), float(weight), int(domsize))
#		rules[rule] = Rule(rule, float(prod), 1.0, int(domsize))
	return rules

def save_rules(rules, filename):
	with open_to_write(filename) as fp:
		for rule in rules.values():
			write_line(fp, (rule.tr, rule.prod, rule.weight, rule.domsize))

def save_lexicon(lexicon, filename):
	with open_to_write(filename) as fp:
		for node in lexicon.values():
			if node.prev is not None:
				rule = None
				for r, w in node.prev.next.iteritems():
					if w == node:
						rule = r
						break
				write_line(fp, (node.prev.word, node.prev.freq, node.word, node.freq, rule))

def save_analyses(lexicon, filename):
	with open_to_write(filename) as fp:
		for word in lexicon.values():
			write_line(fp, (word.word, '<- ' + ' <- '.join(word.analysis())))

def expectation_maximization(corpus, unigrams, iter_count):
	# load rules and add the end-derivation-rule
	rules = load_rules(RULES_FILE + '.' + str(iter_count-1))
	if not rules.has_key(u'#'):
		rules[u'#'] = Rule(u'#', 1.0, 1.0, len(corpus))
	# build lexicon and reestimate parameters
	lexicon = build_lexicon(rules, corpus, unigrams)
	reestimate_rule_prod(rules, lexicon)
	reestimate_rule_weights(rules, lexicon)
	return rules, lexicon

def null_lex_logl(corpus, unigrams):
	logl = 0.0
	n = 1
	for word in corpus.keys():
		logl += math.log(n) + math.log(ngr.word_prob(word, unigrams, 1))
		n += 1
	return logl

def null_corp_logl(corpus):
	total = sum(corpus.values())
	return sum([freq * (math.log(freq) - math.log(total)) for freq in corpus.values()])

def avg_rule_cost(prod, domsize):
	n = int(round(prod * domsize))
	m = domsize - n
	cost = n * math.log(prod)
	if m > 0:
		cost += m * math.log(1.0 - prod)
	return cost / n

def run():
	corpus = Counter.load_from_file(settings.FILES['wordlist'])
	unigrams = ngr.train(settings.FILES['wordlist'], 1)
	unigrams.normalize()
	print 'Null lexicon LogL:', null_lex_logl(corpus, unigrams)
	print 'Null corpus LogL:', null_corp_logl(corpus)
	for i in range(1, NUM_ITERATIONS+1):
		print '\n===   Iteration %d   ===\n' % i
		rules, lexicon = expectation_maximization(corpus, unigrams, i)
		# save results
		save_rules(rules, RULES_FILE + '.' + str(i))
		if i % 50 == 1:
			save_lexicon(lexicon, LEXICON_FILE + '.' + str(i))
			with open(LEXICON_P_FILE + str(i), 'w+') as fp:
				pickle.dump(lexicon, fp)
			save_analyses(lexicon, ANALYSES_FILE + '.' + str(i))

def compute_rule_costs(rules_file, lexicon_file, outfile):
	unigrams = ngr.train(settings.FILES['wordlist'], 1)
	unigrams.normalize()
	sort_file(lexicon_file, key=5)
	costs, value = {}, {}
	for rule, prod, weight, domsize in load_tsv_file(rules_file):
		costs[rule] = avg_rule_cost(float(prod), int(domsize))
	for rule, wordpairs in load_tsv_file_by_key(lexicon_file, 5):
		if not value.has_key(rule):
			value[rule] = 0.0
		for (word_1, freq_1, word_2, freq_2) in wordpairs:
			value[rule] += math.log(ngr.word_prob(word_2, unigrams, 1)) / len(wordpairs)
			print word_2, ngr.word_prob(word_2, unigrams, 1)
	with open_to_write(outfile) as outfp:
		for rule, cost in sorted([(r, c) for r, c in costs.iteritems()],\
				reverse=True, key=lambda x: x[1]):
			if value.has_key(rule):
				write_line(outfp, (rule, cost, value[rule]))

