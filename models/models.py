import algorithms.ngrams
from datastruct.lexicon import *
from datastruct.rules import *
import settings
from utils.printer import *
from utils.stats import NormCache
from collections import defaultdict
import math
import numpy as np
from numpy.random import uniform
import pickle
from scipy.optimize import fsolve
from scipy.special import gammaln
import random
#from scipy.stats import norm

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

class PointGaussianFeature(Feature):
	def __init__(self, dim=1, mean=None, sdev=None):
		self.dim = dim
		if mean is None:
			if self.dim == 1:
				self.mean = 0.0
			else:
				self.mean = np.array([0.0]*self.dim)
		else:
			self.mean = mean
		if sdev is None:
			if self.dim == 1:
				self.sdev = 1.0
			else:
				self.sdev = np.array([1.0]*self.dim)
		else:
			self.sdev = sdev

	def cost(self, values):
		if self.dim == 1:
#			try:
			return -sum(NormCache.logpdf(val, self.mean, self.sdev)\
				for val in values)
#			except ValueError:
#				print(values)
#				print(self.mean, self.sdev)
#				return -sum(norm.logpdf(val, self.mean, self.sdev)\
#					for val in values)
		else:
			return -sum(NormCache.logpdf(val[j], self.mean[j], self.sdev[j])\
				for val in values for j in range(self.dim))

	def weighted_cost(self, values):
		if self.dim == 1:
			try:
				return np.asscalar(\
					-sum(w*NormCache.logpdf(val, self.mean, self.sdev)\
					for val, w in values))
			except ValueError as e:
				print(self.mean)
				print(self.sdev)
				print(values)
				print(list(w*NormCache.logpdf(val, self.mean, self.sdev)\
					for val, w in values))
				print(\
					-sum(w*NormCache.logpdf(val, self.mean, self.sdev)\
					for val, w in values))
				raise e
		else:
			return -sum(w*NormCache.logpdf(val[j], self.mean[j], self.sdev[j])\
				for val, w in values for j in range(self.dim))

	def fit(self, values):
		if not values: return
		values = np.stack(values)
		self.mean = np.mean(values, axis=0)
		if values.shape[0] > 1:
			self.sdev = np.std(values, axis=0)
#		if self.sdev.shape != self.dim and self.sdev.shape != ():
#			print(self.dim)
#			print(self.sdev.shape)
#			print(self.sdev)
#			raise Exception('Bad STD shape')
		if not np.all(self.sdev):
			print(values)
			raise Exception('sdev = 0')

	def weighted_fit(self, values):
		if not values:
			raise Exception('No values to fit')
#		matrix, weights = [], []
		matrix = np.stack(val for val, w in values)
		weights = np.stack(w for val, w in values)
		num_edges = sum(1 for val, w in values if w > 0)
#		for val, w in values:
#			if w > 0:
#				matrix.append([val] if self.dim == 1 else val)
#				weights.append(w)
#		matrix = np.asarray(matrix)
#		weights = np.asarray(weights)
		if num_edges == 0:
			return
#			raise Exception('Weights sum to 0.')
		self.mean = np.average(matrix, axis=0, weights=weights)
#		if self.dim == 1:
#			self.mean = np.asscalar(self.mean)
		if num_edges <= 1:
			return
		if matrix.shape[0] > 1:
			shape = (matrix.shape[0],) if self.dim == 1 else (matrix.shape[0], 1)
			self.sdev = np.average(\
				(matrix - np.tile(self.mean, shape))**2,\
				axis=0, weights=weights)
			if self.dim == 1:
#				print('matrix = ', matrix)
#				print('weights = ', weights)
#				print(self.sdev.shape, self.sdev)
				self.sdev = np.asscalar(self.sdev)
#			self.sdev = np.std(values, axis=0) * len(values) / np.sum(weights)
#		if not np.all(self.sdev):
#			print(values)
#			raise Exception('sdev = 0')
	
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

class PointGaussianGammaGammaFeature(Feature):
	'''Gaussian feature with Gamma priors on mean and variance.'''

	def __init__(self, alpha_1, beta_1, alpha_2, beta_2):
		self.alpha_1 = alpha_1
		self.beta_1 = beta_1
		self.alpha_2 = alpha_2
		self.beta_2 = beta_2
		self.mean = alpha_1 * beta_1
		self.var = alpha_2 * beta_2
	
	### auxiliary methods: likelihood and its derivatives ###
	
	def gamma_logl(self, alpha, beta, value):
		'''Log-likelihood of the Gamma distribution.'''
		return -alpha*np.log(beta) - gammaln(alpha) +\
			(alpha-1)*np.log(value) - beta*value
	
	def statistics(self, values):
		'''Sufficient statistics for the Gaussian distribution.'''
		s_0 = sum(1 for val in values)
		s_1 = sum(val for val in values)
		s_2 = sum(val**2 for val in values)
		return s_0, s_1, s_2
	
	def weighted_statistics(self, values):
		'''Sufficient statistics for the weighted Gaussian distribution.'''
		s_0 = sum(w for val, w in values)
		s_1 = sum(w*val for val, w in values)
		s_2 = sum(w*val**2 for val, w in values)
		return s_0, s_1, s_2

	def logl(self, s_0, s_1, s_2):
		'''Log-likelihood function.'''
		return lambda x: self.gamma_logl(self.alpha_1, self.beta_1, x[0]) +\
			self.gamma_logl(self.alpha_2, self.beta_2, x[1]) -\
			0.5*s_0*np.log(2*np.pi*x[1]) -\
			(0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2) / x[1]
	
	def dlogl_dmean(self, s_0, s_1, s_2):
		'''First derivative of log-likelihood wrt. mean.'''
#		return lambda x: (self.alpha_1-1)/x[0] - self.beta_1 +\
#			(s_1 - s_0*x[0]) / x[1]
		# multiplied by x[1]**2 for better convergence of numeric methods
		return lambda x: (self.alpha_1-1)*x[1] - x[0]*x[1]*self.beta_1 +\
			(s_1 - s_0*x[0]) * x[0]

	def dlogl_dvar(self, s_0, s_1, s_2):
		'''First derivative of log-likelihood wrt. variance.'''
#		return lambda x: (self.alpha_2-1-0.5*s_0)/x[1] - self.beta_2 +\
#			(0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2)/x[1]**2
		# multiplied by x[1]**2 for better convergence of numeric methods
		return lambda x: (self.alpha_2-1-0.5*s_0)*x[1] - self.beta_2*x[1]**2 +\
			(0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2)

	def dlogl_dmean_dmean(self, s_0, s_1, s_2):
		'''Second derivative of log-likelihood wrt. mean.'''
		return lambda x: -(self.alpha_1-1)/x[0]**2 - s_0/x[1]

	def dlogl_dvar_dvar(self, s_0, s_1, s_2):
		'''Second derivative of log-likelihood wrt. variance.'''
		return lambda x: -(self.alpha_2-1-0.5*s_0)/x[1]**2 -\
			(s_2-2*s_1*x[0]+s_0*x[0]**2)/x[1]**3

	def dlogl_dmean_dvar(self, s_0, s_1, s_2):
		'''The mixed second derivative of the log-likelihood.'''
		return lambda x: (s_0*x[0] - s_1) / x[1]**2
	
	def gradient(self, s_0, s_1, s_2):
		'''The gradient of the log-likelihood.'''
		return lambda x: (self.dlogl_dmean(s_0, s_1, s_2)(x),\
			self.dlogl_dvar(s_0, s_1, s_2)(x))
	
	def hessian(self, s_0, s_1, s_2):
		'''The determinant of the Hessian matrix of the log-likelihood.'''
		return lambda x: self.dlogl_dmean_dmean(s_0, s_1, s_2)(x)*\
			self.dlogl_dvar_dvar(s_0, s_1, s_2)(x) - \
			self.dlogl_dmean_dvar(s_0, s_1, s_2)(x)**2
	
	def fit_to_stats(self, s_0, s_1, s_2, max_iter=100000, max_results=10):
		num_iter, num_results = 1, 0
		fun = self.gradient(s_0, s_1, s_2)
		hes = self.hessian(s_0, s_1, s_2)
		mm = self.dlogl_dmean_dmean(s_0, s_1, s_2)
		ll = self.logl(s_0, s_1, s_2)
		result, result_logl = None, None

#		print()
#		print('s_0 =', s_0)
#		print('s_1 =', s_1)
#		print('s_2 =', s_2)
#		print('mean =', s_1/s_0)
#		print('var =', s_2/s_0 - (s_1/s_0)**2)
#		print()

		while num_iter < max_iter and num_results < max_results:
			x = fsolve(fun, (uniform(0, 10), uniform(0, 10)))
			fx = fun(x)
			if x[0] > 0 and x[1] > 0 and fx[0] < 0.001 and fx[1] < 0.001 and\
					hes(x) > 0.001 and mm(x) < -0.001:
				num_results += 1
#				print(num_iter, x, ll(x), hes(x), mm(x))
				if result_logl is None or ll(x) > result_logl:
					result = x
					result_logl = ll(x)
			num_iter += 1
		if result is None:
			mean = s_1/s_0
			var = s_2/s_0 - (s_1/s_0)**2
			print('fit_to_stats: failed to find a result. m=%f, var=%f, s_0=%f' % (mean, var, s_0))
		else:
			self.mean, self.var = result

	### main methods (for export) ###

	def null_cost(self):
		return -self.gamma_logl(self.alpha_1, self.beta_1, self.mean) -\
			self.gamma_logl(self.alpha_2, self.beta_2, self.var)

	def cost(self, values):
		s_0, s_1, s_2 = self.statistics(values)
		return 0.5*s_0*np.log(2*np.pi*self.var) +\
			(0.5*s_2 - s_1*self.mean + 0.5*s_0*self.mean**2) / self.var

	def weighted_cost(self, values):
		s_0, s_1, s_2 = self.weighted_statistics(values)
		return 0.5*s_0*np.log(2*np.pi*self.var) +\
			(0.5*s_2 - s_1*self.mean + 0.5*s_0*self.mean**2) / self.var

	def fit(self, values):
		s_0, s_1, s_2 = self.statistics(values)
		if s_0 <= 0 or s_2/s_0 - (s_1/s_0)**2 <= 0: return
		self.fit_to_stats(s_0, s_1, s_2)

	def weighted_fit(self, values):
		s_0, s_1, s_2 = self.weighted_statistics(values)
		if s_0 <= 0 or s_2/s_0 - (s_1/s_0)**2 <= 0: return
		self.fit_to_stats(s_0, s_1, s_2)

	def num_args(self):
		raise Exception('Not implemented!')

class PointGaussianGaussianGammaFeature(Feature):
	'''Multivariate Gaussian feature with independent coordinates,
	   zero-centered prior on means and Gamma prior on variances.'''

	def __init__(self, dim, var_0, alpha, beta):
		self.dim = dim			# TODO multidimensional!
		self.var_0 = var_0
		self.alpha = alpha
		self.beta = beta
		self.mean = 0
		self.var = alpha*beta
	
	### auxiliary methods: likelihood and its derivatives ###
	
	def gamma_logl(self, alpha, beta, value):
		'''Log-likelihood of the Gamma distribution.'''
		return -alpha*np.log(beta) - gammaln(alpha) +\
			(alpha-1)*np.log(value) - beta*value
	
	def statistics(self, values):
		'''Sufficient statistics for the Gaussian distribution.'''
		s_0 = sum(1 for val in values)
		s_1 = sum(val for val in values)
		s_2 = sum(val**2 for val in values)
		return s_0, s_1, s_2
	
	def weighted_statistics(self, values):
		'''Sufficient statistics for the weighted Gaussian distribution.'''
		s_0 = sum(w for val, w in values)
		s_1 = sum(w*val for val, w in values)
		s_2 = sum(w*val**2 for val, w in values)
		return s_0, s_1, s_2

	def logl(self, s_0, s_1, s_2):
		'''Log-likelihood function.'''
		return lambda x: -0.5*np.log(2*np.pi*self.var_0) -\
			0.5*x[0]**2 / self.var_0 +\
			self.gamma_logl(self.alpha, self.beta, x[1]) -\
			0.5*s_0*np.log(2*np.pi*x[1]) -\
			(0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2) / x[1]
	
	def dlogl_dmean(self, s_0, s_1, s_2):
		'''First derivative of log-likelihood wrt. mean.'''
		# multiplied by x[1]**2 for better convergence of numeric methods
		return lambda x: x[0]/self.var_0*x[1] + (s_1-s_0*x[0])

	def dlogl_dvar(self, s_0, s_1, s_2):
		'''First derivative of log-likelihood wrt. variance.'''
		# multiplied by x[1]**2 for better convergence of numeric methods
		return lambda x: (self.alpha-1-0.5*s_0)*x[1] - self.beta*x[1]**2 +\
			(0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2)

	def dlogl_dmean_dmean(self, s_0, s_1, s_2):
		'''Second derivative of log-likelihood wrt. mean.'''
		return lambda x: 1/self.var_0 - s_0/x[1]

	def dlogl_dvar_dvar(self, s_0, s_1, s_2):
		'''Second derivative of log-likelihood wrt. variance.'''
		return lambda x: -(self.alpha-1-0.5*s_0)/x[1]**2 -\
			(s_2-2*s_1*x[0]+s_0*x[0]**2)/x[1]**3

	def dlogl_dmean_dvar(self, s_0, s_1, s_2):
		'''The mixed second derivative of the log-likelihood.'''
		return lambda x: (s_0*x[0] - s_1) / x[1]**2
	
	def gradient(self, s_0, s_1, s_2):
		'''The gradient of the log-likelihood.'''
		return lambda x: (self.dlogl_dmean(s_0, s_1, s_2)(x),\
			self.dlogl_dvar(s_0, s_1, s_2)(x))
	
	def hessian(self, s_0, s_1, s_2):
		'''The determinant of the Hessian matrix of the log-likelihood.'''
		return lambda x: self.dlogl_dmean_dmean(s_0, s_1, s_2)(x)*\
			self.dlogl_dvar_dvar(s_0, s_1, s_2)(x) - \
			self.dlogl_dmean_dvar(s_0, s_1, s_2)(x)**2
	
	def fit_to_stats(self, s_0, s_1, s_2, max_iter=100000, max_results=10):
		num_iter, num_results = 1, 0
		fun = self.gradient(s_0, s_1, s_2)
		hes = self.hessian(s_0, s_1, s_2)
		mm = self.dlogl_dmean_dmean(s_0, s_1, s_2)
		ll = self.logl(s_0, s_1, s_2)
		result, result_logl = None, None

#		print()
#		print('s_0 =', s_0)
#		print('s_1 =', s_1)
#		print('s_2 =', s_2)
#		print('mean =', s_1/s_0)
#		print('var =', s_2/s_0 - (s_1/s_0)**2)
#		print()

		while num_iter < max_iter and num_results < max_results:
			x = fsolve(fun, (uniform(0, 10), uniform(0, 10)))
			fx = fun(x)
			if x[0] > 0 and x[1] > 0 and fx[0] < 0.001 and fx[1] < 0.001 and\
					hes(x) > 0.001 and mm(x) < -0.001:
				num_results += 1
#				print(num_iter, x, ll(x), hes(x), mm(x))
				if result_logl is None or ll(x) > result_logl:
					result = x
					result_logl = ll(x)
			num_iter += 1
		if result is None:
			mean = s_1/s_0
			var = s_2/s_0 - (s_1/s_0)**2
			print('fit_to_stats: failed to find a result. m=%f, var=%f, s_0=%f' % (mean, var, s_0))
		else:
			self.mean, self.var = result

	### main methods (for export) ###

	def null_cost(self):
		return -0.5*np.log(2*np.pi*self.var_0)-0.5*self.mean**2/self.var_0 -\
			self.gamma_logl(self.alpha, self.beta, self.var) 

	def cost(self, values):
		s_0, s_1, s_2 = self.statistics(values)
		return 0.5*s_0*np.log(2*np.pi*self.var) +\
			(0.5*s_2 - s_1*self.mean + 0.5*s_0*self.mean**2) / self.var

	def weighted_cost(self, values):
		s_0, s_1, s_2 = self.weighted_statistics(values)
		return 0.5*s_0*np.log(2*np.pi*self.var) +\
			(0.5*s_2 - s_1*self.mean + 0.5*s_0*self.mean**2) / self.var

	def fit(self, values):
		s_0, s_1, s_2 = self.statistics(values)
		if s_0 <= 0 or s_2/s_0 - (s_1/s_0)**2 <= 0: return
		self.fit_to_stats(s_0, s_1, s_2)

	def weighted_fit(self, values):
		s_0, s_1, s_2 = self.weighted_statistics(values)
		if s_0 <= 0 or s_2/s_0 - (s_1/s_0)**2 <= 0: return
		self.fit_to_stats(s_0, s_1, s_2)

	def num_args(self):
		raise Exception('Not implemented!')

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
	'''Creates sets of features according to the program configuration.'''

	def __init__(self):
		self.features = ()
		self.weights = ()
	
	def __getitem__(self, idx):
		return self.features[idx]
	
	@staticmethod
	def new_edge_feature_set(domsize):
		result = FeatureSet()
		features = [PointBinomialFeature(domsize)]
		weights = [1.0]
		if settings.WORD_FREQ_WEIGHT > 0.0:
#			features.append(PointGaussianFeature(dim=1))
			features.append(PointGaussianGammaGammaFeature(2, 1, 1, 10))
			weights.append(settings.WORD_FREQ_WEIGHT)
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(PointGaussianFeature(dim=settings.WORD_VEC_DIM))
			weights.append(settings.WORD_VEC_WEIGHT)
		result.features = tuple(features)
		result.weights = tuple(weights)
		return result

	@staticmethod
	def new_root_feature_set():
		result = FeatureSet()
		features = [StringFeature()]
		weights = [1.0]
		if settings.WORD_FREQ_WEIGHT > 0.0:
			features.append(PointExponentialFeature())
			weights.append(settings.WORD_FREQ_WEIGHT)
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(PointGaussianFeature(dim=settings.WORD_VEC_DIM))
			weights.append(settings.WORD_VEC_WEIGHT)
		result.features = tuple(features)
		result.weights = tuple(weights)
		return result

	@staticmethod
	def new_rule_feature_set():
		result = FeatureSet()
		result.features = (StringFeature(),)		# priors on other features ignored
		result.weights = (1.0,)
		return result
	
	def weighted_cost(self, values):
		return sum(w*f.weighted_cost(val) for f, w, val in\
			zip(self.features, self.weights, values))

	def cost(self, values):
		return sum(w*f.cost(val) for f, w, val in\
			zip(self.features, self.weights, values))

	def null_cost(self):
		return sum(w*f.null_cost()\
			for f, w in zip(self.features, self.weights))
	
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
			features.append(list(node.logfreq for node in nodes))
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(list(node.vec for node in nodes))
		return tuple(features)
	
	def extract_features_from_edge(self, edge):
		pass

	def extract_features_from_edges(self, edges):
		features = [list(1 for e in edges)]
		if settings.WORD_FREQ_WEIGHT > 0.0:
			# source-target, because target-source typically negative
			features.append(\
				[e.source.logfreq - e.target.logfreq for e in edges]
			)
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(\
				list(e.target.vec - e.source.vec for e in edges)
			)
		return tuple(features)
	
	def extract_features_from_weighted_edges(self, edges):
		features = [list((1, w) for e, w in edges)]
		if settings.WORD_FREQ_WEIGHT > 0.0:
			features.append(\
				[(e.source.logfreq - e.target.logfreq, w) for e, w in edges]
			)
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(\
				list((e.target.vec - e.source.vec, w) for e, w in edges)
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

#Types of models:
#-	PointModel (parameters set to points)
#-	MarginalModel (parameters integrated out)

