from datastruct.counter import *
from datastruct.rules import Rule
from utils.files import *
from utils.printer import *
import re
import math

class RuleData:
	def __init__(self, rule_str, freq):
		self.rule_str = rule_str
		self.freq = freq
		rule = Rule.from_string(rule_str)
		self.pattern_1 = re.compile(\
			('^' + rule.prefix[0] + '.*' + \
			'.*'.join([a[0] for a in rule.alternations]) + '.*' + rule.suffix[0] + '$')\
			.replace('.*.*', '.*'))
		self.pattern_2 = re.compile(\
			('^' + rule.prefix[1] + '.*' + \
			'.*'.join([a[1] for a in rule.alternations]) + '.*' + rule.suffix[1] + '$')\
			.replace('.*.*', '.*'))
		self.count_1, self.count_2 = 0, 0
#		print rule_str, self.pattern_1.pattern, self.pattern_2.pattern

def load_rules(rules_file):
	rules = []
	for rule_str, freq in load_tsv_file(rules_file):
		rules.append(RuleData(rule_str, freq))
	return rules

def compute_weights(graph_file, word_freq):
	pp = progress_printer(get_file_size(graph_file))
	rules_w1, rules_w2 = Counter(), Counter()
	for word_1, word_2, rule in load_tsv_file(graph_file):
		rules_w1.inc(rule, count=word_freq[word_1])
		rules_w2.inc(rule, count=word_freq[word_2])
		pp.next()
	rules_w = {}
	for r in rules_w1.keys():
		rules_w[r] = min(float(rules_w2[r]) / rules_w1[r], 1.0)
	return rules_w

def rewrite_weights(input_file, output_file, rules_w):
	with open_to_write(output_file) as outfp:
		for rule, prod, weight, domsize in load_tsv_file(input_file):
			prod = float(prod)
			if prod > 0.9:
				prod = 0.9
			write_line(outfp, (rule, prod, rules_w[rule], domsize))

def estimate_prob(word_freq, rules):
	pp = progress_printer(len(word_freq))
	for word, freq in word_freq.iteritems():
		for r in rules:
			if re.match(r.pattern_1, word):
				r.count_1 += 1
			if re.match(r.pattern_2, word):
				r.count_2 += 1
		pp.next()

import random
def rewrite_prob(input_file, output_file, rules):
	rules_d = dict([(r.rule_str, r) for r in rules])
	with open_to_write(output_file) as outfp:
		for rule, prod, weight, domsize in load_tsv_file(input_file):
			write_line(outfp, (rule, random.random(), random.random(), domsize))
#			write_line(outfp, (rule, float(rules_d[rule].freq) / 267210, weight, domsize))

def write_output(output_file, rules, rules_w):
	with open_to_write(output_file) as fp:
		for r in rules:
			if rules_w.has_key(r.rule_str):
				write_line(fp, (r.rule_str, float(r.freq) / r.count_1,\
					rules_w[r.rule_str], r.count_1))

def run():
#	rules = load_rules('s_rul.txt')
	word_freq = Counter.load_from_file('input.txt')
	print 'Computing productivity...'
#	estimate_prob(word_freq, rules)
	print 'Computing weights...'
	rules_w = compute_weights('graph.txt', word_freq)
	rewrite_weights('rules.txt.0', 'rules.txt.00', rules_w)
#	rewrite_prob('rules.txt.0', 'rules.txt.00', rules)
#	write_output('rules.txt.0', rules, rules_w)

