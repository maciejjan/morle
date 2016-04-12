import math
import random
import sys
import time
from operator import itemgetter
import numpy as np
from scipy.special import beta, betaln
from datastruct.lexicon import *
from datastruct.rules import *
from utils.files import *
from utils.printer import *
import settings

# TODO optimizing the rule set
# - instead choosing rules uniformly, choose according to the percentage of graphs without this rule
#   - or the best option: choose according to the subsample size obtained by choosing this rule?
#     (how to compute it quickly?)
# - if the subsample size falls below 1% of the sample size -> resample?

# TODO check which statistic require starting at a null graph

# Beta distribution parameters
ALPHA = 1
BETA = 1

NUM_WARMUP_ITERATIONS = 1000000
NUM_ITERATIONS = 10000000
NUM_RULE_ITERATIONS = 10000
ANNEALING_TEMPERATURE = 50

### STATISTICS ###

class MCMCStatistic:
	def __init__(self, sampler):		raise Exception('Not implemented!')
	def reset(self, sampler):			raise Exception('Not implemented!')
	def update(self, sampler):			raise Exception('Not implemented!')
	def edge_added(self, sampler):		raise Exception('Not implemented!')
	def edge_removed(self, sampler):	raise Exception('Not implemented!')
	def next_iter(self, sampler):		raise Exception('Not implemented!')

class ScalarStatistic(MCMCStatistic):
	def __init__(self, sampler):
		self.reset(sampler)
	
	def reset(self, sampler):
		self.val = 0
		self.last_modified = 0

	def update(self, sampler):
		raise Exception('Not implemented!')

	def edge_added(self, sampler, idx, word_1, word_2, rule):
		raise Exception('Not implemented!')

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		raise Exception('Not implemented!')

	def next_iter(self, sampler):
		pass
	
	def value(self):
		return self.val

class ExpectedLogLikelihoodStatistic(ScalarStatistic):
	def update(self, sampler):
		pass
	
	def edge_added(self, sampler, idx, word_1, word_2, rule):
		pass

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		pass
	
	def next_iter(self, sampler):
		self.val = (self.val * (sampler.num-1) + sampler.logl) / sampler.num

class TimeStatistic(ScalarStatistic):
	def reset(self, sampler):
		self.started = time.time()
		self.val = 0
	
	def update(self, sampler):
		self.val = time.time() - self.started
	
	def edge_added(self, sampler, idx, word_1, word_2, rule):
		pass

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		pass

class AcceptanceRateStatistic(ScalarStatistic):
	def update(self, sampler):
		pass
	
	def edge_added(self, sampler, idx, word_1, word_2, rule):
		self.acceptance(sampler)

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		self.acceptance(sampler)

	def acceptance(self, sampler):
		if sampler.num > self.last_modified:
			self.val = (self.val * self.last_modified + 1) / sampler.num
			self.last_modified = sampler.num

class EdgeStatistic(MCMCStatistic):
	def __init__(self, sampler):
		self.reset(sampler)

	def reset(self, sampler):
		self.values = [0] * sampler.len_edges
		self.last_modified = [0] * sampler.len_edges
	
	def update(self, sampler):
		raise Exception('Not implemented!')

	def edge_added(self, sampler, idx, word_1, word_2, rule):
		raise Exception('Not implemented!')

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		raise Exception('Not implemented!')

	def next_iter(self, sampler):
		pass
	
	def value(self, idx):
		return self.values[idx]

class EdgeFrequencyStatistic(EdgeStatistic):
	def update(self, sampler):
		for i, (word_1, word_2, rule) in enumerate(sampler.edges):
			if sampler.lexicon[word_2].prev == sampler.lexicon[word_1]:
				# the edge was present in the last graphs
				self.edge_removed(sampler, i, word_1, word_2, rule)
			else:
				# the edge was absent in the last graphs
				self.edge_added(sampler, i, word_1, word_2, rule)

	def edge_added(self, sampler, idx, word_1, word_2, rule):
		self.values[idx] =\
			self.values[idx] * self.last_modified[idx] / sampler.num
		self.last_modified[idx] = sampler.num

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		self.values[idx] =\
			(self.values[idx] * self.last_modified[idx] +\
			 (sampler.num - self.last_modified[idx])) /\
			sampler.num
		self.last_modified[idx] = sampler.num

class RuleStatistic(MCMCStatistic):
	def __init__(self, sampler):
		self.values = {}
		self.last_modified = {}
		self.reset(sampler)

	def reset(self, sampler):
		for rule in sampler.lexicon.ruleset.keys():
			self.values[rule] = 0.0
			self.last_modified[rule] = 0
	
	def update(self, sampler):
		for rule in sampler.lexicon.ruleset.keys():
			self.update_rule(rule, sampler)
	
	def update_rule(self, rule, sampler):
		raise Exception('Not implemented!')
	
	def edge_added(self, sampler, idx, word_1, word_2, rule):
		raise Exception('Not implemented!')

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		raise Exception('Not implemented!')

	def next_iter(self, sampler):
		pass
	
	def value(self, rule):
		return self.values[rule]

class RuleFrequencyStatistic(RuleStatistic):
	def update_rule(self, rule, sampler):
		self.values[rule] = \
			(self.values[rule] * self.last_modified[rule] +\
			 sampler.lexicon.rules_c[rule] * (sampler.num - self.last_modified[rule])) /\
			sampler.num
		self.last_modified[rule] = sampler.num
	
	def edge_added(self, sampler, idx, word_1, word_2, rule):
		self.update_rule(rule, sampler)

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		self.update_rule(rule, sampler)

class RuleChangeCountStatistic(RuleStatistic):
	def reset(self, sampler):
		for rule in sampler.lexicon.ruleset.keys():
			self.values[rule] = 0
			self.last_modified[rule] = 0

	def update_rule(self, rule, sampler):
		pass
	
	def edge_added(self, sampler, idx, word_1, word_2, rule):
		if sampler.lexicon.rules_c[rule] == 1:
			self.values[rule] += 1

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		if sampler.lexicon.rules_c[rule] == 0:
			self.values[rule] += 1

class RuleGraphsWithoutStatistic(RuleStatistic):
	def update_rule(self, rule, sampler):
		if sampler.lexicon.rules_c[rule] > 0:
			self.values[rule] = \
				self.values[rule] * self.last_modified[rule] / sampler.num
			self.last_modified[rule] = sampler.num
		else:
			self.values[rule] = \
				(self.values[rule] * self.last_modified[rule] +\
				 sampler.num - self.last_modified[rule]) / sampler.num
	
	def edge_added(self, sampler, idx, word_1, word_2, rule):
		if sampler.lexicon.rules_c[rule] == 1:
			self.values[rule] = \
				(self.values[rule] * self.last_modified[rule] +\
				 sampler.num - self.last_modified[rule]) / sampler.num
			self.last_modified[rule] = sampler.num

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		if sampler.lexicon.rules_c[rule] == 0:
			self.values[rule] = \
				self.values[rule] * self.last_modified[rule] / sampler.num
			self.last_modified[rule] = sampler.num

class RuleIntervalsWithoutStatistic(MCMCStatistic):
	def __init__(self, sampler):
		self.intervals = {}
		self.int_start = {}
		self.reset(sampler)

	def reset(self, sampler):
		for rule in sampler.lexicon.ruleset.keys():
			self.intervals[rule] = []
			if sampler.lexicon.rules_c[rule] > 0:
				self.int_start[rule] = None
			else:
				self.int_start[rule] = 0

	def update(self, sampler):
		for rule in sampler.lexicon.ruleset.keys():
			if self.int_start[rule] is not None:
				self.intervals[rule].append((self.int_start[rule], sampler.num))
				self.int_start[rule] = None
	
	def edge_added(self, sampler, idx, word_1, word_2, rule):
		if sampler.lexicon.rules_c[rule] == 1:
			if self.int_start[rule] is None:
				raise Exception('Interval with no left end: %s' % rule)
			self.intervals[rule].append((self.int_start[rule], sampler.num))
			self.int_start[rule] = None

	def edge_removed(self, sampler, idx, word_1, word_2, rule):
		if sampler.lexicon.rules_c[rule] == 0:
			self.int_start[rule] = sampler.num
	
	def next_iter(self, sampler):
		pass

### SAMPLERS ###

class MCMCGraphSampler:
	def __init__(self, lexicon, edges):
		self.lexicon = lexicon
		self.edges = edges
		self.edges_hash = {}
		for idx, (w1, w2, r) in enumerate(edges):
			self.edges_hash[(w1, w2)] = (idx, r)
		self.len_edges = len(edges)
		self.num = 0		# iteration number
		self.stats = {}
		# precompute some values to speed up simulation
		self.word_prob = {}
		for word in lexicon.keys():
			self.word_prob[word] = lexicon.rootdist.word_prob(word)
		self.logl = sum([math.log(v) for v in self.word_prob.values()]) +\
			sum([math.log(self.lexicon.ruleset.rsp.rule_prob(r.rule)) +\
			     betaln(ALPHA, r.domsize + BETA) - betaln(ALPHA, BETA) \
				for r in self.lexicon.ruleset.values()])
	
	def add_stat(self, name, stat):
		if name in self.stats:
			raise Exception('Duplicate statistic name: %s' % name)
		self.stats[name] = stat

	# TODO check whether the rule is in the current rule set
	def next(self):
		def consider_adding_edge(word_1, word_2, rule):
			return (self.lexicon.rules_c[rule] + ALPHA) /\
			       (self.word_prob[word_2] *\
			        (self.lexicon.ruleset[rule].domsize - self.lexicon.rules_c[rule] - 1 + BETA))

		def consider_removing_edge(word_1, word_2):
			rule = self.edges_hash[(word_1, word_2)][1]
			return self.word_prob[word_2] *\
			       (self.lexicon.ruleset[rule].domsize - self.lexicon.rules_c[rule] + BETA) /\
				   (self.lexicon.rules_c[rule] - 1 + ALPHA)

		self.num += 1

		# select an edge randomly
		edge_idx = random.randrange(self.len_edges)			
		word_1, word_2, rule = self.edges[edge_idx]

		# determine the changes to the lexicon
		edges_to_add, edges_to_remove, prob_ratio = [], [], 1.0
		if self.lexicon[word_2].prev == self.lexicon[word_1]:
			# delete the selected edge
			edges_to_remove.append((word_1, word_2))
			prob_ratio *= consider_removing_edge(word_1, word_2)
		else:
			# add the selected edge
			edges_to_add.append((word_1, word_2, rule))
			prob_ratio *= consider_adding_edge(word_1, word_2, rule)
			if self.lexicon[word_1].has_ancestor(word_2):
				# swap parents
				# determine word_3 and word_4
				word_3 = self.lexicon[word_2].prev.word\
				         if self.lexicon[word_2].prev is not None else None
				w = self.lexicon[word_1]
				word_4 = w.prev.word
				while w.prev != self.lexicon[word_2]: w = w.prev
				word_5 = w.word
				# choose the variant of the swap move and perform it
				if random.random() < 0.5:
					edges_to_remove.append((word_4, word_1))
					prob_ratio *= consider_removing_edge(word_4, word_1)
					if word_3 is not None and (word_3, word_1) in self.edges_hash:
						rule_3_1 = self.edges_hash[(word_3, word_1)][1]
						edges_to_add.append((word_3, word_1, rule_3_1))
						prob_ratio *= consider_adding_edge(word_3, word_1, rule_3_1)
						edges_to_remove.append((word_3, word_2))
						prob_ratio *= consider_removing_edge(word_3, word_2)
					elif word_3 is not None:
						self.num -= 1
						return
				else:
					edges_to_remove.append((word_2, word_5))
					prob_ratio *= consider_removing_edge(word_2, word_5)
					if word_3 is not None and (word_3, word_5) in self.edges_hash:
						rule_3_5 = self.edges_hash[(word_3, word_5)][1]
						edges_to_add.append((word_3, word_5, rule_3_5))
						prob_ratio *= consider_adding_edge(word_3, word_5, rule_3_5)
						edges_to_remove.append((word_3, word_2))
						prob_ratio *= consider_removing_edge(word_3, word_2)
					elif word_3 is not None:
						self.num -= 1
						return
				# check if all necessary edges exist and determine the rules for them
#				edges_to_add.append((word_1, word_2, rule))
			elif self.lexicon[word_2].prev is not None:
				# delete the current parent of word_2
				word_3 = self.lexicon[word_2].prev.word
				edges_to_remove.append((word_3, word_2))
				prob_ratio *= consider_removing_edge(word_3, word_2)

		# accept/reject
		if prob_ratio >= 1 or prob_ratio >= random.random():
			# check whether all edges to add are possible
			for (w_1, w_2, r) in edges_to_add:
				if r in self.lexicon[w_1].next:
					self.num -= 1
					return
			# update log-likelihood
			self.logl += math.log(prob_ratio)
			# remove edges and update stats
			for (w_1, w_2) in edges_to_remove:
#				r = self.lexicon[w_2].deriving_rule()
				idx, r = self.edges_hash[(w_1, w_2)]
				self.lexicon.remove_edge(w_1, w_2)
				for stat in self.stats.values():
					stat.edge_removed(self, idx, w_1, w_2, r)
			# add edges and update stats
			for (w_1, w_2, r) in edges_to_add:
				idx, r = self.edges_hash[(w_1, w_2)]
				self.lexicon.draw_edge(w_1, w_2, r)
				for stat in self.stats.values():
					stat.edge_added(self, idx, w_1, w_2, r)

		# update the remaining stats
		for stat in self.stats.values():
			stat.next_iter(self)
	
	def reset(self):
		self.num = 0
		for stat in self.stats.values():
			stat.reset(self)

	def update_stats(self):
		for stat in self.stats.values():
			stat.update(self)

class MCMCRuleSetSampler:
	#TODO: sample rules: choose a rule to delete and sample graphs to determine acceptance/rejection
	def __init__(self, sample, intervals, ruleset, starting_temp):
		self.sample = sample		# list of log-likelihoods (with a full rule set)
		self.sample_len = len(sample)
		self.intervals = intervals
		self.ruleset = ruleset
		self.rules_list = list(self.ruleset.keys())
		self.len_ruleset = len(ruleset)
		self.deleted_rules = set()
		self.best_set = None					# store the best set of rules
		self.logl = sum(sample)/self.sample_len	# current log-likelihood
		self.starting_temp = starting_temp
		self.temperature = starting_temp
		self.subsample_size = self.sample_len
		self.best_deleted_rules = set()
		self.best_logl = self.logl
		self.num = 0
	
	def next(self, logfp):
		def rule_zero_frequency_prob(rule):
			return beta(ALPHA, self.ruleset[rule].domsize + BETA) / beta(ALPHA, BETA)

		def merge_two_intervals(int_1, int_2):
			i, j = 0, 0
			result = []
			if not int_1 or not int_2: return result
			a_1, b_1 = int_1[0]
			a_2, b_2 = int_2[0]
			while i < len(int_1) and j < len(int_2):
				if b_1 <= a_2:
					i += 1
					if i == len(int_1): break
					a_1, b_1 = int_1[i]
					continue
				elif b_2 <= a_1:
					j += 1
					if j == len(int_2): break
					a_2, b_2 = int_2[j]
					continue
				else:
					if b_1 > b_2:
						result.append((max(a_1, a_2), b_2))
						j += 1
						if j == len(int_2): break
						a_2, b_2 = int_2[j]
						continue
					else:
						result.append((max(a_1, a_2), b_1))
						if b_1 == b_2:
							j += 1
							if j == len(int_2): break
							a_2, b_2 = int_2[j]
						i += 1
						if i == len(int_1): break
						a_1, b_1 = int_1[i]
						continue
			return result

		def merge_n_intervals(ints):
			result = [(0, self.sample_len)]
			for interval in ints:
				result = merge_two_intervals(result, interval)
			return result

		# TODO what if deleted_rules is empty?
		def likelihood(deleted_rules):
			# compute likelihood with the set of deleted rules from the sample
			result, sum_weights = 0.0, 0.0
			weight, log_weight = 1.0, 0.0
			log_prior_weight = 0.0
			for rule in deleted_rules:
				rzfp = rule_zero_frequency_prob(rule)
#				weight /= rzfp
				log_weight += math.log(rzfp)
				log_prior_weight += math.log(self.ruleset.rsp.rule_prob(rule))
#			print(weight)
			intervals = merge_n_intervals([self.intervals[r] for r in deleted_rules])
			if not intervals:
				return -float('inf')
			for int_begin, int_end in intervals:
				for logl in self.sample[int_begin:int_end]:
					result += logl - log_weight - log_prior_weight
					sum_weights += 1
#			if sum_weights < 0.01 * NUM_ITERATIONS:
#				return -float('inf')
			result /= sum_weights
			self.subsample_size = int(sum_weights)
			return result

		self.num += 1
		# choose the rule to change (add/delete) at random
		rule = self.rules_list[random.randrange(self.len_ruleset)]
		# compute the likelihoods with/without the rule
		new_logl = likelihood(self.deleted_rules ^ {rule})
#		print(new_logl, self.logl, new_logl-self.logl, self.temperature, (new_logl-self.logl) / self.temperature)
		acc_prob = 1.0 if new_logl > self.logl else math.exp((new_logl-self.logl) / self.temperature)
		# accept with probability exp(dh/T) (dh - likelihood change)
		if acc_prob >= 1 or acc_prob >= random.random():
			if rule in self.deleted_rules:
				logfp.write('restoring: '+rule+' num_rules=' + str(self.len_ruleset-len(self.deleted_rules))+\
				            ' subsample_size='+str(self.subsample_size)+' old_logl='+str(self.logl)+\
							' new_logl='+str(new_logl)+' best_logl='+str(self.best_logl)+'\n')
			else:
				logfp.write('deleting: '+rule+' num_rules=' + str(self.len_ruleset-len(self.deleted_rules))+\
				            ' subsample_size='+str(self.subsample_size)+' old_logl='+str(self.logl)+\
							' new_logl='+str(new_logl)+' best_logl='+str(self.best_logl)+'\n')
			self.deleted_rules ^= {rule}
			self.logl = new_logl
			if self.logl > self.best_logl:
				self.best_deleted_rules = self.deleted_rules
				self.best_logl = self.logl
		# T decreases as G/log(iter_num) for some G
		self.temperature = self.starting_temp / np.log2(self.num+1)
	
	def save_best_ruleset(self, filename):
		rs = RuleSet()
		for r in self.ruleset.values():
			if not r.rule in self.best_deleted_rules:
				rs[r.rule] = r
		rs.save_to_file(filename)

### AUXILIARY FUNCTIONS ###

def optimize_rules(lexicon, edges, outfile):
	deleted_rules = set()
	iter_num = 0
	while True:
		iter_num += 1
		print()
		print('======== ITERATION %d =========' % iter_num)
		print()
		print('edges: %d' % len(edges))
		print('rules: %d' % len(lexicon.ruleset))
		lexicon.reset()
		sampler = MCMCGraphSampler(lexicon, edges)
		sampler.add_stat('int_without', RuleIntervalsWithoutStatistic(sampler))
		print('Warming up the chain...')
		pp = progress_printer(NUM_WARMUP_ITERATIONS)
		while sampler.num < NUM_WARMUP_ITERATIONS:
			num = sampler.num
			sampler.next()
			if sampler.num > num: next(pp)
		sampler.reset()
		print('Sampling graphs...')
		sample = np.zeros(NUM_ITERATIONS)
		pp = progress_printer(NUM_ITERATIONS)
		while sampler.num < NUM_ITERATIONS:
			num = sampler.num
			sampler.next()
			if sampler.num > num:
				next(pp)
				sample[sampler.num-1] = sampler.logl
		sampler.update_stats()
		print('Sampling rules...')
		rule_sampler = MCMCRuleSetSampler(sample, sampler.stats['int_without'].intervals,\
			sampler.lexicon.ruleset, ANNEALING_TEMPERATURE)
		with open_to_write('logs/sampling-rules.log.'+str(iter_num)) as logfp:
			pp = progress_printer(NUM_RULE_ITERATIONS)
			while rule_sampler.num < NUM_RULE_ITERATIONS:
				next(pp)
				rule_sampler.next(logfp)
		if rule_sampler.best_deleted_rules == deleted_rules:
			rule_sampler.save_best_ruleset(outfile)
			break
		else:
			deleted_rules = rule_sampler.best_deleted_rules
			edges = list(filter(lambda e: e[2] not in deleted_rules, edges))
			for rule in deleted_rules:
				del sampler.lexicon.ruleset[rule]

def load_edges(filename):
	return list(read_tsv_file(filename, (str, str, str)))

def save_edges(sampler, filename):
	stats, stat_names = [], []
	for stat_name, stat in sorted(sampler.stats.items(), key = itemgetter(0)):
		if isinstance(stat, EdgeStatistic):
			stat_names.append(stat_name)
			stats.append(stat)
	with open_to_write(filename) as fp:
		write_line(fp, ('word_1', 'word_2', 'rule') + tuple(stat_names))
		for i, (word_1, word_2, rule) in enumerate(sampler.edges):
			write_line(fp, (word_1, word_2, rule) + tuple([stat.value(i) for stat in stats]))

def save_rules(sampler, filename):
	stats, stat_names = [], []
	for stat_name, stat in sorted(sampler.stats.items(), key = itemgetter(0)):
		if isinstance(stat, RuleStatistic):
			stat_names.append(stat_name)
			stats.append(stat)
	with open_to_write(filename) as fp:
		write_line(fp, ('rule', 'domsize') + tuple(stat_names))
		for rule in sampler.lexicon.ruleset.keys():
			r = sampler.lexicon.ruleset[rule]
			write_line(fp, (rule, r.domsize) +\
			               tuple([stat.value(rule) for stat in stats]))

def print_scalar_stats(sampler):
	stats, stat_names = [], []
	print()
	print()
	print('SIMULATION STATISTICS')
	print()
	spacing = max([len(stat_name)\
	               for stat_name, stat in sampler.stats.items() 
					   if isinstance(stat, ScalarStatistic)]) + 2
	for stat_name, stat in sorted(sampler.stats.items(), key = itemgetter(0)):
		if isinstance(stat, ScalarStatistic):
			print((' ' * (spacing-len(stat_name)))+stat_name, ':', stat.value())
	print()
	print()

def save_intervals(intervals, filename):
	with open_to_write(filename) as fp:
		for rule, ints in intervals.items():
			write_line(fp, (rule, len(ints), ' '.join([str(i) for i in ints])))

