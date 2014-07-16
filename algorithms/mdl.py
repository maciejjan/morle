import algorithms.ngrams
import algorithms.optrules
from datastruct.lexicon import *
from datastruct.rules import *
from utils.files import *
import settings

GRAPH_MDL_FILE = 'graph_mdl.txt'
GAMMA_THRESHOLD = 1e-20

# TODO:
# - integrate optrules into the module
#	- patterns, make_rule etc. -> datastruct.rules
# - settings (e.g. filenames)
#   - lexicon.training
#   - wordlist.training
#   - wordlist.testing
#   - lexicon.gs
# - switch: supervised / unsupervised
# - switch: use frequencies or not (if not -- ignore corpus probability)
# - integrate word generation (mdl-analyze) into the module

def build_lexicon(rules, lexicon):
	print 'Resetting lexicon...'
	lexicon.reset()

	# compute the improvement for each possible edge
	word_best_edge = {}
	for word_1, word_2, rule in read_tsv_file(settings.FILES['surface.graph'],\
			print_progress=False):
#			print_progress=True, print_msg='Computing edge scores...'):
		if rules.has_key(rule):
			delta_logl = lexicon.try_edge(word_1, word_2, rules[rule])
			if delta_logl < 0:
				continue
			if not word_best_edge.has_key(word_2) or word_best_edge[word_2] < delta_logl:
				word_best_edge[word_2] = delta_logl

	# words should be processed in the order of the best edge score,
	#   so sort them according to this value
	word_num = dict([(word, num) for num, (word, delta_logl) in enumerate(\
		sorted(word_best_edge.iteritems(), reverse=True, key=lambda x:x[1]), 1)])
	# write the desired position for each word into the graph
	num_lines = 0
	with open_to_write(GRAPH_MDL_FILE) as outfp:
		for word_1, word_2, rule in read_tsv_file(settings.FILES['surface.graph'],\
				print_progress=False):
#				print_progress=True, print_msg='Writing edge scores...'):
			if word_num.has_key(word_2) and rules.has_key(rule):
				write_line(outfp, (word_1, word_2, rule, word_num[word_2]))
				num_lines += 1
	set_file_size(GRAPH_MDL_FILE, num_lines)
	# sort the graph according to desired positions of words
	sort_file(GRAPH_MDL_FILE, numeric=True, key=4)

	# for each word, compute once more the improvement for each possible edge
	#   and draw the best edges
	for word_2, rest in read_tsv_file_by_key(GRAPH_MDL_FILE, 2,\
				print_progress=False):
#				print_progress=True, print_msg='Building lexicon...'):
		edges = sorted([(word_1, word_2, rule,\
			lexicon.try_edge(word_1, word_2, rules[rule]))\
			for (word_1, rule, _) in rest],\
			reverse=True, key = lambda x: x[3])
		for edge in edges:
			word_1, word_2, rule, delta_logl = edge
			if delta_logl > 0 and not word_2 in lexicon[word_1].analysis()\
					and not lexicon[word_1].next.has_key(rule):
#				print word_1, word_2, rule, delta_logl, len(lexicon.roots)
				lexicon.draw_edge(word_1, word_2, rules[rule])
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
			if rules[r].prod > 0.9:
				rules[r].prod = 0.9
		elif r == u'#':
			pass
		else:
			del rules[r]

def reestimate_rule_weights(rules, lexicon):
	# gradient descent
	rules_w = dict([(r.rule, r.weight) for r in rules.values()])
	old_logl = lexicon.corpus_logl(rules)
	while True:
		d = lexicon.logl_gradient(rules)
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
			new_logl = lexicon.try_weights(new_rules_w)
			if new_logl > old_logl + 100.0:
				print 'improvement', new_logl, 'over', old_logl, 'with gamma =', gamma
				rules_w = new_rules_w
				old_logl = new_logl
				# TODO update weights here!
				break
			else:
#				print 'no improvement', new_logl, old_logl
				gamma /= 10
		if gamma < GAMMA_THRESHOLD:
			break
	print lexicon.corpus_logl(rules)
	# normalize the rule weight
	print rules_w[u'#']
	for r in rules.values():
		r.weight = rules_w[r.rule] / rules_w[u'#']
	# update weight sums in lexicon nodes
	for n in lexicon.values():
		n.sum_weights = 1.0 + sum([rules[r].weight for r in n.next.keys()])
	print lexicon.corpus_logl(rules)

def load_training_file(filename):
	rules, lexicon = RuleSet(), Lexicon()
	unigrams = algorithms.ngrams.NGramModel(1)
	words, total = [], 0
	for word, freq in read_tsv_file(filename, (unicode, int)):
		words.append((word, freq))
		lexicon[word] = LexiconNode(word, freq, freq, 0.0, 0.0, 1.0)
		lexicon.roots.add(word)
		lexicon.total += freq
	unigrams.train(words)
	for word_2, freq, word_1 in read_tsv_file(filename, (unicode, int, unicode), print_progress=True):
		lexicon[word_2].ngram_prob = unigrams.word_prob(word_2)
		if word_1 != u'-':
			if not lexicon.has_key(word_1):
				lexicon[word_1] = LexiconNode(word_1, 0, 0, unigrams.word_prob(word_1), 0.0, 1.0)
				lexicon.roots.add(word_1)
			rule = algorithms.align.align(word_1, word_2).to_string()
			if not rules.has_key(rule):
				rules[rule] = RuleData(rule, 1.0, 1.0, 0)
			lexicon.draw_edge(word_1, word_2, rules[rule], corpus_prob=False)
	# compute corpus probabilities
	for w in lexicon.values():
		w.corpus_prob = float(w.sigma) / lexicon.total * w.sum_weights
	return unigrams, rules, lexicon
	
