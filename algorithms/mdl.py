import algorithms.branching
import algorithms.ngrams
import algorithms.optrules
import algorithms.align
from algorithms.optrules import extract_all_rules
from datastruct.lexicon import *
from datastruct.rules import *
from utils.files import *
from utils.printer import *
import settings
import re

import math

def check_rules(lexicon):
	'''For each rule, compute the approximate drop in log-likelihood
	   caused by deleting this rule. Delete rules that are not worth maintaining.'''
	gain, analyzed, dependent = {}, set(), set()
	edges = []
	for word_1, freq_1, word_2, freq_2, rule, weight, delta_logl, d1, d2 in\
			read_tsv_file('edges.txt',\
			(str, int, str, int, str, int, float, float, float),\
			print_progress=True, print_msg='Checking rules...'):
		edges.append((word_1, word_2, rule, delta_logl))
	edges.sort(reverse=True, key=lambda x: x[3])
	for word_1, word_2, rule, delta_logl in edges:
		if rule not in lexicon.ruleset:
			continue
		if lexicon[word_2].prev == lexicon[word_1]:
			if word_2 not in gain:
				gain[word_2] = 0.0
			gain[word_2] += delta_logl
		elif word_2 not in analyzed\
				and word_2 not in lexicon[word_1].analysis()\
				and rule not in lexicon[word_1].next:
			if word_2 not in gain:
				gain[word_2] = 0.0
			gain[word_2] -= delta_logl
			analyzed.add(word_2)
			dependent.add(rule)
	# save word gain to a file and calculate gain for rules
	with open_to_write('word_gain.txt') as fp:
		rules_gain = {}
		for w in lexicon.values():
			for rule, w2 in w.next.items():
				if rule not in rules_gain:
					rules_gain[rule] = 0.0
				rules_gain[rule] += gain[w2.word]
				write_line(fp, (w2.word, gain[w2.word], rule))
	# compute rule score and filter rules
	rules_score = []
	for rule, g in rules_gain.items():
#		cost = math.log(lexicon.ruleset.rule_cost(rule, lexicon.rules_c[rule])
		r = lexicon.ruleset[rule]
		cost = math.log(lexicon.ruleset.rsp.rule_prob(rule)) +\
			(r.domsize-lexicon.rules_c[rule]) * math.log(1-r.prod)
		rules_score.append((rule, g, cost, cost+g))
#		if cost+g < 0.0 and rule not in dependent:
		if cost+g < 0.0:
			del lexicon.ruleset[rule]
	rules_score.sort(key=lambda x: x[3])
	with open_to_write('rules_gain.txt') as fp:
		for r, g, c, sc in rules_score:
			write_line(fp, (r, g, c, sc))

def optimize_lexicon(lexicon):
	print('Resetting lexicon...')
	lexicon.reset()

	# delete rules with frequency class < 0
#	lexicon.ruleset.filter(lambda r: r.freqcl >= 0)
	for r in lexicon.ruleset.values():
		if r.freqcl < 0:
			r.freqcl = 0

	# compute the improvement for each possible edge
	vertices, edges = set(), []
	with open_to_write('edges.txt') as outfp:
		for word_1, word_2, rule in read_tsv_file(settings.FILES['surface.graph'],\
				print_progress=True, print_msg='Computing edge scores...'):
			if rule in lexicon.ruleset:
				delta_logl = lexicon.try_edge(word_1, word_2, rule)
				delta_cor_logl = math.log(lexicon.ruleset[rule].freqprob(lexicon[word_2].freqcl - lexicon[word_1].freqcl))
				write_line(outfp, (word_1, lexicon[word_1].freqcl, word_2, lexicon[word_2].freqcl,\
					rule, int(lexicon.ruleset[rule].freqcl), 
					'{:.4}'.format(delta_logl), 
					'{:.4}'.format(delta_logl-delta_cor_logl), 
					'{:.4}'.format(delta_cor_logl)))
				if delta_logl < 0:
					continue
				edges.append((word_1, word_2, rule, delta_logl))
			if word_1 not in vertices:
				vertices.add(word_1)
			if word_2 not in vertices:
				vertices.add(word_2)
	edges.sort(reverse=True, key=lambda x: x[3])
	update_file_size('edges.txt')

	print('Computing maximum branching...')
	branching = algorithms.branching.branching(list(vertices), edges)
	for v1, v2, rule in branching:
		if rule not in lexicon[v1].next:
			lexicon.draw_edge(v1, v2, rule)

# TODO: build_lexicon_iter?
# start with the outermost "layer", then proceed to deeper layers

def optimize_rule_params(lexicon):
	print('Optimizing rule parameters...')
	lexicon.ruleset.filter(lambda r: r.rule in lexicon.rules_c and\
	                                 lexicon.rules_c[r.rule] > 0)
	for rule, r in lexicon.ruleset.items():
		r.prod = min(lexicon.rules_c[rule] / r.domsize, settings.MAX_PROD)
	if settings.USE_WORD_FREQ:
		for r in lexicon.ruleset.values():
			r.freqcl = 0
		for w in lexicon.values():
			if w.prev is not None:
				lexicon.ruleset[w.deriving_rule()].freqcl += w.freqcl - w.prev.freqcl
		for r in lexicon.ruleset.values():
			r.freqcl //= lexicon.rules_c[r.rule]

######################################################


#def find_new_ancestor(lexicon, rules, root, depth=0):
#	node = lexicon[root]
#	max_depth = 0
#	for i in range(depth):
#		if i in node.structure:
#			max_depth += 1
#		else:
#			break
#	if node.prev and max_depth < depth:
#		if max_depth > 0:
#			new_ancestor = node.prev
#			for i in range(depth-max_depth):
#				new_ancestor = new_ancestor.prev
#			lexicon.remove_edge(node.prev.word, node.word)
#			if new_ancestor is not None:
#				max_prod, rule = 0.0, None
#				for r in algorithms.optrules.extract_all_rules(new_ancestor.word, node.word):
#					if rules.has_key(r) and rules[r].prod > max_prod:
#						rule = r
#						max_prod = rules[r].prod
#				if rule is not None:
#					lexicon.draw_edge(new_ancestor.word, node.word, rules[rule])
#				else:
#					rule = algorithms.align.align(new_ancestor.word, node.word)
#					r = rule.to_string()
#					domsize = 0
#					for word in lexicon.keys():
#						if rule.lmatch(word):
#							domsize += 1
#					rules[r] = RuleData(r, 1.0 / domsize, node.freq - new_ancestor.freq, domsize)
##					print 'Added rule:', r, node.freq - new_ancestor.freq, domsize
#					lexicon.draw_edge(new_ancestor.word, node.word, rules[r])
#			node.annotate_word_structure(max_depth)
#	for child in node.next.values():
#		find_new_ancestor(lexicon, rules, child.word, max_depth+1)
#
#def rebuild_lexicon(lexicon, rules):
#	print('Rebuilding lexicon...')
#	pp = progress_printer(len(lexicon.roots))
#	for root in list(lexicon.roots):
#		rebuild_tree(lexicon, rules, root)
#		pp.next()
#
#def rebuild_tree(lexicon, rules, root):
#	# for each node:
#	#   annotate word structure (1=root etc.)
#	#     for each letter at each position: look for it by ancestors
#	lexicon[root].annotate_word_structure()
#	find_new_ancestor(lexicon, rules, root)
#	#   determine possible ancestors
#	#   hang the word over and extract all rules possible for this edge
#	#   if some rules in the rule set -> choose the best, else add the most general one
#
##def rebuild_tree(lexicon, rules, root):
##	words = lexicon[root].words_in_tree()
##	substrings = {}
##	for w1, w2, r in read_tsv_file(settings.FILES['surface.graph']):
##		if w1 in words and w2 in words:
##			if not substrings.has_key(w1):
##				substrings[w1] = []
##			if not substrings.has_key(w2):
##				substrings[w2] = []
##			substrings[w2].append((w1, r, len(algorithms.align.lcs(w1, w2)), lexicon[w1].depth()))
##	words.sort(reverse=True, key=lambda x: lexicon[x].depth())
##	for w in words:
##		lexicon[w].prev = None
##		lexicon[w].next = {}
##		lexicon.roots.add(w)
##	for w in words:
##		max_wp, max_r, max_lcs, min_dep = None, None, 0, 0
##		for wp, r, lcs, dep in substrings[w]:
##			if not w in lexicon[wp].analysis() and not lexicon[wp].next.has_key(r) and rules.has_key(r):
##				if lcs > max_lcs:
##					max_wp = wp
##					max_r = r
##					max_lcs = lcs
##					min_dep = dep
##				elif lcs == max_lcs and dep < min_dep:
##					max_wp = wp
##					max_r = r
##					min_dep = dep
##		if max_wp is not None:
##			lexicon.draw_edge(max_wp, w, rules[max_r])

def load_training_file_with_freq(filename):
	ruleset = RuleSet()
	rootdist = algorithms.ngrams.NGramModel(1)
	rootdist.train_from_file(filename)
	lexicon = Lexicon(rootdist=rootdist, ruleset=ruleset)

	for word, freq in read_tsv_file(filename, (str, int)):
		lexicon[word] = LexiconNode(word, freq, rootdist.word_prob(word))
		lexicon.roots.add(word)
	for word_2, freq, word_1 in read_tsv_file(filename, (str, int, str),\
			print_progress=True, print_msg='Building lexicon from training data...'):
		if word_1 != u'-' and word_1 != word_2 and word_2 in lexicon.roots:
			if word_1 not in lexicon:
				lexicon.add_word(word_1, 1, rootdist.word_prob(word_1))
			rule = algorithms.align.align(word_1, word_2).to_string()
			if rule not in ruleset:
				ruleset[rule] = RuleData(rule, 1.0, 1, 0)
			lexicon.draw_edge(word_1, word_2, rule)
	return lexicon

def load_training_file_without_freq(filename):
	ruleset = RuleSet()
	rootdist = algorithms.ngrams.NGramModel(1)
	rootdist.train([(word, 1) for (word,) in read_tsv_file(filename, (str,))])
	lexicon = Lexicon(rootdist=rootdist, ruleset=ruleset)

	for (word,) in read_tsv_file(filename, (str,)):
		lexicon[word] = LexiconNode(word, 1, rootdist.word_prob(word))
		lexicon.roots.add(word)
	for word_2, word_1 in read_tsv_file(filename, (str, str),\
			print_progress=True, print_msg='Building lexicon from training data...'):
		if word_1 != u'-' and word_1 != word_2 and word_2 in lexicon.roots:
			if word_1 not in lexicon:
				lexicon.add_word(word_1, 1, rootdist.word_prob(word_1))
			rule = algorithms.align.align(word_1, word_2).to_string()
			if rule not in ruleset:
				ruleset[rule] = RuleData(rule, 1.0, 1, 0)
			lexicon.draw_edge(word_1, word_2, rule)
	return lexicon

#	rules, lexicon = RuleSet(), Lexicon()
#	unigrams = algorithms.ngrams.NGramModel(1)
#	words, total = [], 0
#	for word in read_tsv_file(filename, (unicode)):
#		words.append((word, 1))
#		lexicon[word] = LexiconNode(word, 0, 0, 0.0, 0.0, 1.0)
#		lexicon.roots.add(word)
#	unigrams.train(words)
#	for word_2, word_1 in read_tsv_file(filename, (unicode, unicode),\
#			print_progress=True, print_msg='Building lexicon from training data...'):
#		lexicon[word_2].ngram_prob = unigrams.word_prob(word_2)
#		if word_1 != u'-' and word_1 != word_2:
#			if not lexicon.has_key(word_1):
#				lexicon[word_1] = LexiconNode(word_1, 0, 0, unigrams.word_prob(word_1), 0.0, 1.0)
#				lexicon.roots.add(word_1)
#			rule = algorithms.align.align(word_1, word_2).to_string()
#			if not rules.has_key(rule):
#				rules[rule] = RuleData(rule, 1.0, 1.0, 0)
#			lexicon.draw_edge(word_1, word_2, rules[rule], corpus_prob=False)
#	return unigrams, rules, lexicon

#TODO supervised
def load_training_file(filename):
	if settings.USE_WORD_FREQ:
		return load_training_file_with_freq(filename)
	else:
		return load_training_file_without_freq(filename)
	
