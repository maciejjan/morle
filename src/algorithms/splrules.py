from algorithms.align import extract_all_rules
from datastruct.rules import *
from collections import defaultdict
from operator import itemgetter
import libhfst
import sys

def compute_possible_edges(edges):
	rules, possible_edges, rules_c = {}, defaultdict(lambda: list()), defaultdict(lambda: 0)
	for e in edges:
		for rule in extract_all_rules(e.source, e.target):
			possible_edges[str(rule)].append(e)
			rules_c[str(rule)] += 1
			if str(rule) not in rules:
				rules[str(rule)] = rule
	for rule_str, count in rules_c.items():
		if count <= 1 or len(edges) - count == 1:
			del possible_edges[rule_str]
			del rules[rule_str]
	return possible_edges, rules

# TODO name subfunctions
#def split_rule_rec(root_rule, real_edges, trh, domsizes, model):
def split_rule_rec(root_rule, real_edges, lexicon, domsizes, model):
	if len(real_edges[root_rule]) <= 1:
		return
	gains = []
	possible_edges, rules = compute_possible_edges(real_edges[root_rule])
	if root_rule not in rules:
		rules[root_rule] = Rule.from_string(root_rule)
#		raise Exception('%s' % root_rule)
	for rule_str in rules:
		if not rule_str in domsizes:
#			domsizes[rule_str] = rules[rule_str].compute_domsize(trh)
			domsizes[rule_str] = rules[rule_str].compute_domsize(lexicon)
	for rule_str, edges in possible_edges.items():
		edges_set = set(edges)
		if rules[rule_str] <= rules[root_rule]:
			remaining_edges = [e for e in real_edges[root_rule]\
				if e not in edges_set]
			try:
				gain = model.rule_split_gain(
					rules[rule_str], edges, domsizes[rule_str],
					rules[root_rule], remaining_edges, domsizes[root_rule]
				)
				gains.append((rule_str, gain))
			except Exception:
				for e in edges+remaining_edges:
					print(str(e.source), str(e.target), str(e.rule))
				print(rule_str, len(edges), domsizes[rule_str])
				print(root_rule, len(remaining_edges), domsizes[root_rule])
				raise Exception()
	if gains:
		best_rule, best_gain = max(gains, key=itemgetter(1))
		if best_gain > 0.0:	# TODO target: 0.0
			real_edges[best_rule] = possible_edges[best_rule]
			real_edges[root_rule] = [e for e in real_edges[root_rule]\
				if e not in possible_edges[best_rule]]
			split_rule_rec(best_rule, real_edges, lexicon, domsizes, model)
#			split_rule_rec(best_rule, real_edges, trh, domsizes, model)
			split_rule_rec(root_rule, real_edges, lexicon, domsizes, model)
#			split_rule_rec(root_rule, real_edges, trh, domsizes, model)

#def split_rule(rule_str, edges, trh, model, domsizes=None):
def split_rule(rule_str, edges, lexicon, model, domsizes=None):
	if domsizes is None:
		domsizes = {}
	real_edges = {rule_str : set(edges)}
#	split_rule_rec(rule_str, real_edges, trh, domsizes, model)
	split_rule_rec(rule_str, real_edges, lexicon, domsizes, model)
	rules = {}
	result = []
	for r_str, edges in real_edges.items():
		rules[r_str] = Rule.from_string(r_str)
		for e in edges:
			e.rule = rules[r_str]
			result.append(e)
	# TODO return edges by rule and domsizes
	return result

