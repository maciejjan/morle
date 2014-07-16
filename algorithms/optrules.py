from datastruct.counter import *
from datastruct.rules import Rule
from datastruct.lexicon import *
from utils.files import *
from utils.printer import *
import algorithms.align
import math
import re

MAX_ALTERNATION_LENGTH = 3
MAX_AFFIX_LENGTH = 5
IMPROVEMENT_THRESHOLD = 50.0

def max_prod_ratio(x):
	return x / (x+1) * math.exp(-math.log(x+1)/x)

def make_rule(alignment):
	pref = alignment[0]
	alt = []
	for i, (x, y) in enumerate(alignment[1:-1], 1):
		if x != y:
			alt.append((x, y))
	suf = alignment[-1]
	return Rule(pref, alt, suf)

def fix_alignment(al):
	if al[0] == (u'', u'') and al[1][0] != al[1][1]:
		al = al[1:]
	if al[-1] == (u'', u'') and al[-2][0] != al[-2][1]:
		al = al[:-1]
	if len(al) == 1:
		al.append((u'', u''))
	return al

def process_alignments(alignments):
	rules_tree = {}
	for i, al in enumerate(alignments):
		rule = make_rule(al)
		new_alignments = []
		for j in range(len(al)-1):
			x1, y1, x2, y2 = al[j][0], al[j][1], al[j+1][0], al[j+1][1]
			k = 2
			if not x2 and not y2 and j < len(al)-2:
				x2, y2 = al[j+2][0], al[j+2][1]
				k = 3
			max_len = MAX_AFFIX_LENGTH if j == 0 or j+k >= len(al)\
				else MAX_ALTERNATION_LENGTH
			if len(x1)+len(x2) <= max_len and \
					len(y1)+len(y2) <= max_len:
				if (x1 or y1) and (x2 != y2):
					new_al = al[:j] + [(x1+x2, y1+y2)] + al[j+k:]
					if not new_al in alignments and not new_al in new_alignments:
						new_alignments.append(fix_alignment(new_al))
				if (x1 != y1) and (x2 or y2):
					new_al = al[:j] + [(x1+x2, y1+y2)] + al[j+k:]
					if not new_al in alignments and not new_al in new_alignments:
						new_alignments.append(fix_alignment(new_al))
		if new_alignments:
			return alignments[:i+1] +\
				process_alignments(alignments[i+1:] + new_alignments)
	return alignments

def extract_all_rules(word_1, word_2):
	cs = algorithms.align.lcs(word_1, word_2)
	pattern = re.compile('(.*)' + '(.*?)'.join([\
		letter for letter in cs]) + '(.*)')
	m1 = pattern.search(word_1)
	m2 = pattern.search(word_2)
	alignment = []
	for i, (x, y) in enumerate(zip(m1.groups(), m2.groups())):
		alignment.append((x, y))
		if i < len(cs):
			alignment.append((cs[i], cs[i]))
	alignments = process_alignments([alignment])
	rules = []
	for al in alignments:
		rules.append(make_rule(al).to_string())
	return set(rules)

def improvement_fun(r, n1, m1, n2, m2):
	n3, m3 = n1-n2, m1
	result = 0.0
	result -= n1 * (math.log(n1) - math.log(m1))
	if m1-n1 > 0: result -= (m1-n1) * (math.log(m1-n1) - math.log(m1))
	result += n2 * (math.log(n2) - math.log(m2))
	if m2-n2 > 0: result += (m2-n2) * (math.log(m2-n2) - math.log(m2))
	if n3 > 0: result += n3 * (math.log(n3) - math.log(m3))
	if m3-n3 > 0: result += (m3-n3) * (math.log(m3-n3) - math.log(m3))
	return result

def optimize_rule(rule, wordpairs, wordlist, matching=None):
	if not wordpairs:
		return []
	# generate candidate rules and count to how many wordpairs they apply
#	print 'Generating candidate rules...'
	applying = Counter()
#	pp = progress_printer(len(wordpairs))
	for word_1, word_2 in wordpairs:
		for r in extract_all_rules(word_1, word_2):
			applying.inc(r)
#		pp.next()
	# remove rules with frequency 1
	for r in applying.keys():
		if applying[r] == 1:
			del applying.entries[r]
			applying.total -= 1
	# TODO better handling of rules with ambigous LCS
	if not applying.has_key(rule):
		applying.inc(rule, count=len(wordpairs))
	rules = {}
	for r in applying.keys():
		rules[r] = Rule.from_string(r)
	# count how many words match the constraints of each rule
#	print 'Matching words against rule patterns...'
	if matching is None:
		matching = Counter()
#		pp = progress_printer(len(wordlist))
		for word in wordlist:
			for r, r_obj in rules.iteritems():
				if r_obj.lmatch(word):
					matching.inc(r)
#			pp.next()
	# compute the improvement brought by isolating each rule
	rule_counts = [(r, improvement_fun(r, applying[rule], matching[rule],\
		applying[r], matching[r])) for r in applying.keys() if matching.has_key(r)]
	if not rule_counts:
		return []

#	print rule, sorted(rule_counts, reverse=True, key=lambda x:x[1])
#	if rule == u':/e:u/:g':
#		print ''
#		for r, imp in sorted(rule_counts, reverse=True, key=lambda x: x[1]):
#			if matching.has_key(r):
#				print r, applying[rule], matching[rule], applying[r], matching[r], imp
	optimal_rule, improvement = max(rule_counts, key = lambda x: x[1])
	if improvement > IMPROVEMENT_THRESHOLD:
		wordpairs_new, wordpairs_rest = [], []
		for w1, w2 in wordpairs:
			if rules[optimal_rule].lmatch(w1):
				wordpairs_new.append((w1, w2))
			else:
				wordpairs_rest.append((w1, w2))
		return optimize_rule(optimal_rule, wordpairs_new, wordlist, matching) + \
			optimize_rule(rule, wordpairs_rest, wordlist, matching)
	else:
		return [(rule, applying[rule], matching[rule], wordpairs)]

def optimize_rules_in_graph(wordlist_file, input_file, output_file, rules_file):
#	sort_file(input_file, key=3)
	wordlist = []
	for word, freq in read_tsv_file(wordlist_file):
		wordlist.append(word)
	with open_to_write(output_file) as outfp:
		with open_to_write(rules_file) as rulesfp:
			for rule, wordpairs in read_tsv_file_by_key(input_file, 3,\
					print_progress=True, print_msg='Optimizing rules in the graph...'):
				opt = optimize_rule(rule, wordpairs, wordlist)
				for new_rule, freq, domsize, new_wordpairs in opt:
					for nw1, nw2 in new_wordpairs:
						write_line(outfp, (nw1, nw2, new_rule))
					prod = float(freq) / domsize
					if prod > 0.9:
						prod = 0.9
					write_line(rulesfp, (new_rule, prod, 1.0, domsize))

def optimize_rules_in_lexicon(input_file, output_file, rules_file):
	lexicon = Lexicon.load_from_file(input_file)
	wordlist = lexicon.keys()
	sort_file(input_file, key=3)
	with open_to_write(rules_file) as rulesfp:
		for rule, rest in read_tsv_file_by_key(input_file, 3,\
				print_progress=True, print_msg='Optimizing rules in the lexicon...'):
			if not rule:
				continue
			wordpairs = [(r[0], r[1]) for r in rest]
			opt = optimize_rule(rule, wordpairs, wordlist)
			for new_rule, freq, domsize, new_wordpairs in opt:
				for nw1, nw2 in new_wordpairs:
					del lexicon[nw1].next[rule]
					lexicon[nw1].next[new_rule] = lexicon[nw2]
				write_line(rulesfp, (new_rule, float(freq)/domsize, 1.0, domsize))
	lexicon.save_to_file(output_file)

