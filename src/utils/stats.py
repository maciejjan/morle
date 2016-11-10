import numpy as np
from scipy.stats import norm

class NormCache:
	'''Cached values of the standard normal distribution PDF'''
	cache_pdf = None
	cache_logpdf = None
	min = None
	max = None
	step = None
	
	def init_cache(min, max, step):
		NormCache.min = min
		NormCache.max = max
		NormCache.step = step
		NormCache.cache_pdf = np.empty(int((max-min)/step))
		NormCache.cache_logpdf = np.empty(int((max-min)/step))
		for i, x in enumerate(np.arange(min, max, step)):
			NormCache.cache_pdf[i] = norm.pdf(x)
			NormCache.cache_logpdf[i] = norm.logpdf(x)

	def pdf(x, mean=0, sdev=1):
		if NormCache.cache_pdf is None:
			NormCache.init_cache(-10, 10, 0.0001)
		x_std = (x-mean)/sdev
		idx = int(np.round((x_std-NormCache.min)/NormCache.step))
		if 0 < idx < NormCache.cache_pdf.shape[0]:
			return NormCache.cache_pdf[idx]
		else:
			return norm.pdf(x_std)
	
	def logpdf(x, mean=0, sdev=1):
		if NormCache.cache_logpdf is None:
			NormCache.init_cache(-10, 10, 0.0001)
		x_std = (x-mean)/sdev
		idx = int(np.round((x_std-NormCache.min)/NormCache.step))
		if 0 < idx < NormCache.cache_logpdf.shape[0]:
			return NormCache.cache_logpdf[idx]
		else:
			return norm.pdf(x_std)
	
