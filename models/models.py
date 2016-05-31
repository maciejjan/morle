import algorithms.ngrams
from datastruct.lexicon import *
from datastruct.rules import *
import settings
from utils.printer import *
from collections import defaultdict
import math
import numpy as np
import pickle
import random
from scipy.stats import norm

class Feature:
	def null_cost(self):
		return 0.0

# a string feature drawn from an n-gram distribution
class StringFeature(Feature):
	def __init__(self):
		self.log_probs = {}

	def cost(self, values):
		return sum(self.smoothing if ngram not in self.log_probs\
			else self.log_probs[ngram]\
			for val in values for ngram in val)

	def fit(self, values):
		counts, total = defaultdict(lambda: 0), 0
		for value in values:
			for ngram in value:
				counts[ngram] += 1
				total += 1
		self.log_probs = {}
		self.smoothing = -math.log(1 / total)
		for ngram, count in counts.items():
			self.log_probs[ngram] = -math.log(count / total)
	
	def num_args(self):
		return 1
	
	def parse_string_args(self, log_probs):
		pass

class PointBinomialFeature(Feature):
	def __init__(self, trials, prob=0.5):
		self.trials = trials
		self.prob = prob
	
	def cost(self, values):
		value = sum(values)
		if (self.prob == 0.0 and value == 0) or\
				(self.prob == 1.0 and value == self.trials):
			return 0.0
		elif self.prob <= 0.0 or self.prob >= 1.0:
			raise Exception('Impossible event: %f %d %d ' % (self.prob, value, self.trials))
		return -value*(math.log(self.prob)-math.log(1-self.prob))
	
	def weighted_cost(self, values):
		value = sum(val*w for val, w in values)
		if (self.prob == 0.0 and value == 0) or\
				(self.prob == 1.0 and value == self.trials):
			return 0.0
		elif self.prob <= 0.0 or self.prob >= 1.0:
			raise Exception('Impossible event: %f %d %d ' % (self.prob, value, self.trials))
		return -value*(math.log(self.prob)-math.log(1-self.prob))
	
	# the normalizing constant of the distribution
	def null_cost(self):
		return -self.trials*math.log(1-self.prob)
	
	def update(self, prob):
		self.prob = prob
	
	def fit(self, values):
		self.prob = len(values) / self.trials
		self.prob = min(max(self.prob, 1e-10), 0.9999)
	
	def weighted_fit(self, values):
		self.prob = sum(val*w for val, w in values) / self.trials
		self.prob = min(max(self.prob, 1e-10), 0.9999)
	
	def num_args(self):
		return 2
	
	def parse_string_args(self, trials, prob):
		self.trials = int(trials)
		self.prob = float(prob)

# TODO dimensions?
class PointGaussianFeature(Feature):
	def __init__(self, dim=1, mean=0.0, sdev=1.0):
		self.dim = dim
		self.mean = mean
		self.sdev = sdev

	def cost(self, values):
		if self.dim == 1:
			try:
				return -sum(norm.logpdf(val, self.mean, self.sdev)\
					for val in values)
			except ValueError:
				print(values)
				print(self.mean, self.sdev)
				return -sum(norm.logpdf(val, self.mean, self.sdev)\
					for val in values)
		else:
			return -sum(norm.logpdf(val[j], self.mean[j], self.sdev[j])\
				for val in values for j in range(self.dim))

	# TODO add/remove edges

	def fit(self, values):
		if not values: return
		values = np.asarray(list(values))
		self.mean = np.mean(values, axis=0)
		if values.shape[0] > 1:
			self.sdev = np.std(values, axis=0)
		if self.sdev == 0:
			print(values)
			raise Exception('sdev = 0')
	
	def weighted_fit(self, values, weights):
		raise Exception('Not implemented!')
	
	def num_args(self):
		return 2
	
	def parse_string_args(self, mean, sdev):
		pass

class PointExponentialFeature(Feature):
	def __init__(self, rate=1.0):
		self.rate = rate

	def cost(self, values):
		return -sum(math.log(self.rate) + self.rate*val for val in values)

	# TODO add/remove edges

	def fit(self, values):
		if not values: return
		values = np.asarray(list(values))
		self.rate = 1.0 / np.mean(values)

class MarginalBinomialFeature(Feature):
	def __init__(self, trials, count):
		self.trials = trials
		self.count = count
#		self.cost = -math.log(p) # TODO
	
	def log_prob(self, value):
		return self.cost
	
	def update(self, count):
		self.count = count

class MarginalGaussianFeature(Feature):
	pass	# parameters for the normal-gamma distribution

class FeatureSet:
	'''Creates sets of features according to programconfiguration.'''

	def __init__(self):
		self.features = ()
	
	def __getitem__(self, idx):
		return self.features[idx]
	
	@staticmethod
	def new_edge_feature_set(domsize):
		result = FeatureSet()
		features = [PointBinomialFeature(domsize)]
		if settings.WORD_FREQ_WEIGHT > 0.0:
			features.append(PointGaussianFeature(dim=1))
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(PointGaussianFeature(dim=settings.WORD_VEC_DIM))
		result.features = tuple(features)
		return result

	@staticmethod
	def new_root_feature_set():
		result = FeatureSet()
		features = [StringFeature()]
		if settings.WORD_FREQ_WEIGHT > 0.0:
			features.append(PointExponentialFeature())
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(PointGaussianFeature(dim=settings.WORD_VEC_DIM))
		result.features = tuple(features)
		return result

	@staticmethod
	def new_rule_feature_set():
		result = FeatureSet()
		result.features = (StringFeature(),)		# priors on other features ignored
		return result
	
	def weighted_cost(self, values):
		return sum(f.weighted_cost(val) for f, val in zip(self.features, values))

	def cost(self, values):
		return sum(f.cost(val) for f, val in zip(self.features, values))

	def null_cost(self):
		return sum(f.null_cost() for f in self.features)
	
	def fit(self, values):
		for feature, f_val in zip(self.features, values):
			feature.fit(f_val)

	def weighted_fit(self, values):
		for feature, f_val in zip(self.features, values):
			feature.weighted_fit(f_val)

class FeatureExtractor:
	def __init__(self):
		pass

	def extract_features_from_nodes(self, nodes):
		features = []
		features.append(algorithms.ngrams.generate_n_grams(\
			node.word + node.tag + ('#',), settings.ROOTDIST_N)\
				for node in nodes)
		if settings.WORD_FREQ_WEIGHT > 0.0:
			features.append(node.logfreq for node in nodes)
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(node.vec for node in nodes)
		return tuple(features)
	
	def extract_features_from_edge(self, edge):
		pass

	def extract_features_from_edges(self, edges):
		features = [list(1 for e in edges)]
		if settings.WORD_FREQ_WEIGHT > 0.0:
			features.append(\
				[e.target.logfreq - e.source.logfreq for e in edges]
			)
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(\
				[e.target.vec - e.source.vec for e in edges]
			)
		return tuple(features)
	
	def extract_features_from_weighted_edges(self, edges):
		features = [list((1, w) for e, w in edges)]
		if settings.WORD_FREQ_WEIGHT > 0.0:
			features.append(\
				[(e.target.logfreq - e.source.logfreq, w) for e, w in edges]
			)
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(\
				[(e.target.vec - e.source.vec, w) for e, w in edges]
			)
		return tuple(features)

	def extract_features_from_rules(self, rules):
		ngrams = []
		for (rule, features) in rules:
			ngrams.append(algorithms.ngrams.generate_n_grams(\
				rule.seq() + ('#',), 1))
		return (tuple(ngrams),)


#models.py - assign probabilities/log-probabilities to objects
#-	hyperparameters
#	(root distributions etc.)
#-	edge_gain
#-	rule_cost (prior)
#-	word_cost (prior)
#-	cost of a rule with all its edges? (total_rule_cost)
#-	rule_split_gain  (~ improvement_fun from "optrules")
#-	fit parameters to the lexicon (roots distrib.) and rule set
#-	etc.

class Model:
	def save_to_file(self, filename):
		with open(settings.WORKING_DIR + filename, 'w+b') as fp:
#			pickle.Pickler(fp, encoding=settings.ENCODING).dump(self)
			pickle.Pickler(fp).dump(self)

	@staticmethod
	def load_from_file(filename):
		with open(settings.WORKING_DIR + filename, 'rb') as fp:
#			pickle.Unpickler(fp, encoding=settings.ENCODING).load()
			return pickle.Unpickler(fp).load()
	
#	def compute_rule_domsize(self):
#		def id_alpha(id_num):
#			ALPHABET = 'abcdefghijklmnoprstuvwxyz'
#			result = ALPHABET[id_num % len(ALPHABET)]
#			while id_num > len(ALPHABET):
#				id_num //= len(ALPHABET)
#				result += ALPHABET[id_num % len(ALPHABET)]
#			return result

class PointModel(Model):	# TODO parallelize
	def __init__(self, lexicon=None, rules=None, edges=None):
		self.extractor = FeatureExtractor()
		self.word_prior = None
		self.rule_prior = None
#		self.rules = set()
		self.rule_features = {}
		self.cost = 0.0
	
	def add_rule(self, rule, rule_features):
		self.rule_features[rule] = rule_features
		self.cost += self.rule_cost(rule)
#		self.cost += self.rule_cost(rule) +\
#			self.rule_features[rule].null_cost()
	
	def delete_rule(self, rule):
		self.cost -= self.rule_cost(rule)
#		self.cost -= self.rule_cost(rule) +\
#			self.rule_features[rule].null_cost()
		del self.rule_features[rule]
	
	def num_rules(self):
		return len(self.rule_features)
	
	def has_rule(self, rule):
		if isinstance(rule, Rule):
			return rule in self.rule_features
		elif isinstance(rule, str):
			return Rule.from_string(rule) in self.rule_features
	
#	def prior_cost(self):
#		pass
	
	def fit_word_prior(self, lexicon):
		self.word_prior = FeatureSet.new_root_feature_set()
		self.word_prior.fit(
			self.extractor.extract_features_from_nodes(lexicon.values()))
	
	def fit_rule_prior(self, rule_features=None):
#		if rules is None:
#			rules = self.rules
		if rule_features is None:
			rule_features = self.rule_features
		self.rule_prior = FeatureSet.new_rule_feature_set()
		self.rule_prior.fit(
			self.extractor.extract_features_from_rules(
				(rule, features)\
					for rule, features in rule_features.items()))
	
	def fit_rule(self, rule, edges, domsize):
		features = FeatureSet.new_edge_feature_set(domsize)
		features.fit(self.extractor.extract_features_from_edges(edges))
		return features
	
	def weighted_fit_rule(self, rule, edges, domsize):
		features = FeatureSet.new_edge_feature_set(domsize)
		features.weighted_fit(\
			self.extractor.extract_features_from_weighted_edges(edges))
		return features
	
	# TODO
#	def fit_rules(self, edges_by_rule, domsizes):
#		#self.rules, self.rule_features = {}, {}
#		self.rule_features = {}
#		for rule, edges in edges_by_rule.items():
#			self.fit_rule(rule, edges, domsizes[rule])
	
	def fit_to_lexicon(self, lexicon):
		for rule, edges in lexicon.edges_by_rule.items():
			domsize = self.rule_features[rule][0].trials
			features = self.fit_rule(rule, edges, domsize)
			old_cost = self.cost
#			print(self.rule_cost(rule), self.rule_features[rule].null_cost())
			self.delete_rule(rule)
			# delete the rule if it brings more loss than gain -- TODO to a separate function
			rule_gain = features.cost(\
				self.extractor.extract_features_from_edges(edges)) -\
				self.rule_cost(rule) -\
				features.null_cost() +\
				sum(e.target.cost for e in edges)
#			print(str(rule), len(edges), rule_gain)
			if rule_gain <= 0.0:
				pass
#				self.add_rule(rule, features)
#				print('Deleting rule: %s gain=%f' % (str(rule), rule_gain))
			else:
				self.add_rule(rule, features)
#				if self.cost - old_cost >= 0.01:
#					print(str(rule), old_cost, self.cost)
#					print(self.rule_cost(rule), features.null_cost())
#					raise Exception('!')
#				del self.rules[rule]
#				del self.rule_features[rule]
		# delete the rules with no edges
		rules_to_delete = [r for r in self.rule_features if r not in lexicon.edges_by_rule]
		for r in rules_to_delete:
#			print('Deleting rule: %s (no edges)' % str(r))
			self.delete_rule(r)
#			del self.rules[r]
#			del self.rule_features[r]

	def fit_to_sample(self, edges_by_rule):
		for rule, edges in edges_by_rule.items():
			domsize = self.rule_features[rule][0].trials
			features = self.weighted_fit_rule(rule, edges, domsize)
			self.delete_rule(rule)
			# delete the rule if it brings more loss than gain (TODO weighted)
			rule_gain = features.weighted_cost(\
				self.extractor.extract_features_from_weighted_edges(edges)) -\
				self.rule_cost(rule) -\
				features.null_cost() +\
				sum(e.target.cost*weight for e, weight in edges)
			if rule_gain <= 0.0:
#				print('Deleting rule: %s gain=%f' % (str(rule), rule_gain))
#				self.delete_rule(rule)
				pass
			else:
				self.add_rule(rule, features)
		# delete the rules with no edges (TODO check whether necessary?)
		rules_to_delete = [r for r in self.rule_features if r not in edges_by_rule]
		for r in rules_to_delete:
#			print('Deleting rule: %s (no edges)' % str(r))
			self.delete_rule(rule)
#			pass

	def rule_split_gain(self, rule_1, edges_1, domsize_1,\
			rule_2, edges_2, domsize_2):
		edges_3 = edges_1 + edges_2
		features_1 = FeatureSet.new_edge_feature_set(domsize_1)
		features_2 = FeatureSet.new_edge_feature_set(domsize_2)
		features_3 = FeatureSet.new_edge_feature_set(domsize_2)
		values_1 = self.extractor.extract_features_from_edges(edges_1)
		values_2 = self.extractor.extract_features_from_edges(edges_2)
		values_3 = self.extractor.extract_features_from_edges(edges_3)
		features_1.fit(values_1)
		features_2.fit(values_2)
		features_3.fit(values_3)
		return features_3.cost(values_3) - features_1.cost(values_1) -\
			features_2.cost(values_2) - self.rule_cost(rule_1) +\
			(self.rule_cost(rule_2) if len(edges_2) == 0 else 0) +\
			features_3.null_cost() - features_1.null_cost() -\
			features_2.null_cost()
	
	def null_cost(self):
		return sum(features.null_cost()\
			for features in self.rule_features.values())

	def node_cost(self, node):
		return self.word_prior.cost(\
			self.extractor.extract_features_from_nodes((node,)))
	
	# TODO cost of a single edge is not computed properly!
	# -> binomial cost of all edges is computed
	# how to compute the cost of a single edge? (sometimes Bernoulli, sometimes binomial)
	def edge_cost(self, edge):
		return self.rule_features[edge.rule].cost(\
			self.extractor.extract_features_from_edges((edge,)))

	def weighted_edge_cost(self, edge, weight):
		return self.rule_features[edge.rule].weighted_cost(\
			self.extractor.extract_features_from_weighted_edges((edge, weight)))
	
	def rule_cost(self, rule, rule_features=None):
		if rule_features is None:
			if rule in self.rule_features:
				rule_features = self.rule_features[rule]
			else:
				rule_features = ()
		return self.rule_prior.cost(\
			self.extractor.extract_features_from_rules(
				((rule, rule_features),)))
	
	def save_to_file(self, filename):
		# forget the transducers, because they are not serializable
		for rule in self.rule_features:
			rule.transducer = None
		Model.save_to_file(self, filename)
	
class MarginalModel(Model):
	def __init__(self, lexicon, ruleset):
		raise Exception('Not implemented!')

#def fit_model_to_graph(lexicon, trh, graph_file):
def fit_model_to_graph(lexicon, graph_file):
	model = PointModel()
	model.fit_word_prior(lexicon)
	rule_features = {}
	for rule_str, wordpairs in read_tsv_file_by_key(graph_file, key=3,\
			print_progress=True):
		rule = Rule.from_string(rule_str)
		edges = [LexiconEdge(lexicon[w1], lexicon[w2], rule)\
			for w1, w2 in wordpairs]
#		domsize = rule.compute_domsize(trh)
		domsize = rule.compute_domsize(lexicon)
		rule_features[rule] = model.fit_rule(rule, edges, domsize)
#		model.add_rule(rule, rule_featuers[rule])
	model.fit_rule_prior(rule_features)
	for rule, features in rule_features.items():
		model.add_rule(rule, features)
	return model

#-	compute edge costs in advance

#Types of models:
#-	PointModel (parameters set to points)
#-	MarginalModel (parameters integrated out)

# rule data in each of the models:
# - PointModel:
#   - productivity, domsize
#   - if freq. used: freq. multiplier mean and sd
#   - if vec. used: vec. mean and sd
# - MarginalModel:
#   - (exp.) count, domsize
#   - if freq. used: freq. multiplier hyperparameters (mu, kappa, alpha, beta)
#   - if vec. used: vec. hyperparameters (mu, kappa, alpha, beta)

# edge_cost requires rule data as input

# models store all parameters and statistics necessary to compute the costs!
# (maybe even the lexicon and rules themselves?)
# models are notified about changes in the lexicon (edge added/removed)
# 

#def improvement_fun(r, n1, m1, n2, m2):
#	n3, m3 = n1-n2, m1
#	result = 0.0
#	result -= n1 * (math.log(n1) - math.log(m1))
#	if m1-n1 > 0: result -= (m1-n1) * (math.log(m1-n1) - math.log(m1))
#	result += n2 * (math.log(n2) - math.log(m2))
#	if m2-n2 > 0: result += (m2-n2) * (math.log(m2-n2) - math.log(m2))
#	if n3 > 0: result += n3 * (math.log(n3) - math.log(m3))
#	if m3-n3 > 0: result += (m3-n3) * (math.log(m3-n3) - math.log(m3))
#	return result
