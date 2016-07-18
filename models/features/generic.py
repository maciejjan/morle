from collections import defaultdict

import math

class Feature:
	pass

class StringFeature(Feature):
	'''A string feature drawn from an n-gram distribution.'''

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

class FeatureSet:
	pass
