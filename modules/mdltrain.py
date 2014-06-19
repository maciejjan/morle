from datastruct.lexicon import *
from datastruct.rules import *
from utils.files import *
import settings

NUM_ITERATIONS = 9
GAMMA_THRESHOLD = 1e-30
GRAPH_MDL_FILE = 'graph_mdl.txt'
RULES_FILE = 'rules.txt'
LEXICON_FILE = 'lexicon.txt'
ANALYSES_FILE = 'analyses.txt'

def build_lexicon(rules, lexicon):
	print 'Resetting lexicon...'
	lexicon.reset()

	# compute the improvement for each possible edge
	word_best_edge = {}
	for word_1, word_2, rule in read_tsv_file(settings.FILES['surface.graph'],\
			print_progress=True, print_msg='Computing edge scores...'):
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
	with open_to_write(GRAPH_MDL_FILE) as outfp:
		for word_1, word_2, rule in read_tsv_file(settings.FILES['surface.graph'],\
				print_progress=True, print_msg='Writing edge scores...'):
			if word_num.has_key(word_2) and rules.has_key(rule):
				write_line(outfp, (word_1, word_2, rule, word_num[word_2]))
	# sort the graph according to desired positions of words
	sort_file(GRAPH_MDL_FILE, numeric=True, key=4)

	# for each word, compute once more the improvement for each possible edge
	#   and draw the best edges
	for word_2, rest in read_tsv_file_by_key(GRAPH_MDL_FILE, 2,\
				print_progress=True, print_msg='Building lexicon...'):
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
			if new_logl > old_logl + 10.0:
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

def expectation_maximization(lexicon, iter_count):
	# load rules and add the end-derivation-rule
	rules = RuleSet.load_from_file(RULES_FILE + '.' + str(iter_count-1))
	if not rules.has_key(u'#'):
		rules[u'#'] = RuleData(u'#', 1.0, 1.0, len(lexicon))
	print 'LogL =', lexicon.logl(rules)
	# build lexicon and reestimate parameters
	lexicon = build_lexicon(rules, lexicon)
	reestimate_rule_prod(rules, lexicon)
	reestimate_rule_weights(rules, lexicon)
	return rules, lexicon

def save_analyses(lexicon, filename):
	with open_to_write(filename) as fp:
		for word in lexicon.values():
			write_line(fp, (word.word, '<- ' + ' <- '.join(word.analysis())))

def run():
	lexicon = Lexicon.init_from_file(settings.FILES['wordlist'])
	for i in range(1, NUM_ITERATIONS+1):
		print '\n===   Iteration %d   ===\n' % i
		rules, lexicon = expectation_maximization(lexicon, i)
		# save results
		rules.save_to_file(RULES_FILE + '.' + str(i))
		lexicon.save_to_file(LEXICON_FILE + '.' + str(i))
		save_analyses(lexicon, ANALYSES_FILE + '.' + str(i))

