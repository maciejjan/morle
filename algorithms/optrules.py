from algorithms.ngrams import TrigramHash, generate_n_grams
from datastruct.counter import *
from datastruct.rules import *
from datastruct.lexicon import *
from utils.files import *
from utils.printer import *
import algorithms.align
import settings
import math
import re

MAX_ALTERNATION_LENGTH = 3
MAX_AFFIX_LENGTH = 5

def make_rule(alignment, tag):
	pref = alignment[0]
	alt = []
	for i, (x, y) in enumerate(alignment[1:-1], 1):
		if x != y:
			alt.append((x, y))
	suf = alignment[-1]
	return Rule(pref, alt, suf, tag)

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
#		rule = make_rule(al)
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
#		if new_alignments:
#			return alignments[:i+1] +\
#				process_alignments(alignments[i+1:] + new_alignments)
	if new_alignments:
		return process_alignments(alignments + new_alignments)
	else:
		return alignments

def extract_all_rules(word_1, word_2):
	tag = None
	if settings.USE_TAGS:
		p1, p2 = word_1.rfind(u'_'), word_2.rfind(u'_')
		tag = (word_1[p1+1:], word_2[p2+1:])
		word_1, word_2 = word_1[:p1], word_2[:p2]
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
		rules.append(make_rule(al, tag).to_string())
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

def rule_domsize(rule, trh):
    rule = Rule.from_string(rule)
    trigrams = rule.get_trigrams()
    if not trigrams:
        return len(trh)
    words = trh.retrieve(trigrams[0])
    for tr in trigrams[1:]:
        words &= trh.retrieve(tr)
    if settings.USE_TAGS:
        words &= trh.retrieve_tag(rule.tag[0])
    if max([len(rule.prefix[0]), len(rule.suffix[0])]) <= 2 and\
        (not rule.alternations or max([len(x) for (x, y) in rule.alternations]) <= 3):
        return len(words)
    else:
        return len([w for w in words if rule.lmatch(w)])

def optimize_rule(rule, wordpairs, trh, rsp, matching=None):
	if not wordpairs:
		return []
	# generate candidate rules and count to how many wordpairs they apply
	applying = Counter()
	for word_1, word_2 in wordpairs:
		for r in extract_all_rules(word_1, word_2):
			applying.inc(r)
#	# remove rules with frequency 1
	applying.filter(lambda x: x > 1)
	# TODO better handling of rules with ambigous LCS
	if rule not in applying:
		applying.inc(rule, count=len(wordpairs))
	# count how many words match the constraints of each rule
	if matching is None:
		matching = Counter()
	for r in applying.keys():
		if r not in matching:
			matching[r] = rule_domsize(r, trh)
	# compute the improvement brought by isolating each rule
	rule_counts = [(r, improvement_fun(r, applying[rule], matching[rule],\
		applying[r], matching[r])) for r in applying.keys() if r in matching]
	if not rule_counts:
		return []

	optimal_rule, improvement = max(rule_counts, key = lambda x: x[1])
	optimal_rule_obj = Rule.from_string(optimal_rule)
	if improvement + math.log(rsp.rule_prob(rule)) > 20.0:
		wordpairs_new, wordpairs_rest = [], []
		for w1, w2 in wordpairs:
			if optimal_rule_obj.lmatch(w1):
				wordpairs_new.append((w1, w2))
			else:
				wordpairs_rest.append((w1, w2))
		return optimize_rule(optimal_rule, wordpairs_new, trh, rsp, matching) + \
			optimize_rule(rule, wordpairs_rest, trh, rsp, matching)
#		return optimize_rule(optimal_rule, wordpairs_new, trh, rsp, matching) + \
#			optimize_rule(rule, wordpairs_rest, trh, rsp, matching) + \
#			[(rule, applying[rule], matching[rule], wordpairs)]
	else:
		return [(rule, applying[rule], matching[rule], wordpairs)]

def optimize_rules_in_graph(wordlist_file, input_file, output_file, rules):
	'''Optimize rules in a graph of patterns.'''
	# load the wordlist into a trigram hash
	trh = TrigramHash()
	word_freqcl, max_freq = {}, None
	for word, freq in read_tsv_file(wordlist_file, (str, int)):
		trh.add(word)
		if max_freq is None:
			max_freq = freq
		word_freqcl[word] = freqcl(freq, max_freq)
	# optimize rules
	with open_to_write(output_file) as outfp:
		for rule, wordpairs in read_tsv_file_by_key(input_file, 3,
				print_progress=True, print_msg='Optimizing rules in the graph...'):
			rule_freqcl = sum([word_freqcl[w2]-word_freqcl[w1] for w1, w2 in wordpairs]) //\
				len(wordpairs)
#			if rule_freqcl < 0:		# discard rules with frequency class < 0
#				continue
			opt = optimize_rule(rule, wordpairs, trh, rules.rsp)
			for new_rule, freq, domsize, new_wordpairs in opt:
				for nw1, nw2 in new_wordpairs:
					write_line(outfp, (nw1, nw2, new_rule))
				prod = min(freq / domsize, 0.9)
				nr_freqcl = sum([word_freqcl[w2]-word_freqcl[w1] for w1, w2 in new_wordpairs]) //\
					len(new_wordpairs)
				rules[new_rule] = RuleData(new_rule, prod, nr_freqcl, domsize)

def optimize_rules_in_lexicon(input_file, output_file, rules_file):
	ruleset = RuleSet.load_from_file(rules_file)
	lexicon = Lexicon.load_from_file(input_file, ruleset)
	trh = TrigramHash()
	for word in lexicon.keys():
		trh.add(word)

	sort_file(input_file, key=3)
	for rule, rest in read_tsv_file_by_key(input_file, 3,\
			print_progress=True, print_msg='Optimizing rules in the lexicon...'):
		if not rule:
			continue
		del lexicon.ruleset[rule]
		wordpairs = [(r[0], r[1]) for r in rest]
		opt = optimize_rule(rule, wordpairs, trh, ruleset.rsp)
		for new_rule, freq, domsize, new_wordpairs in opt:
			for nw1, nw2 in new_wordpairs:
				del lexicon[nw1].next[rule]
				lexicon[nw1].next[new_rule] = lexicon[nw2]
			lexicon.ruleset[new_rule] = RuleData(new_rule, freq/domsize, 0, domsize)
#				write_line(rulesfp, (new_rule, float(freq)/domsize, 0, domsize))
	lexicon.save_model(rules_file, output_file)

