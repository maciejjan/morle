import algorithms.ngrams
import settings

class FeatureValueExtractor:
	def __init__(self):
		pass

	def extract_feature_values_from_nodes(self, nodes):
		features = []
		features.append(list(algorithms.ngrams.generate_n_grams(\
			node.word + node.tag + ('#',), settings.ROOTDIST_N)\
				for node in nodes))
		if settings.WORD_FREQ_WEIGHT > 0.0:
			features.append(list(node.logfreq for node in nodes))
		if settings.WORD_VEC_WEIGHT > 0.0:
			features.append(list(node.vec for node in nodes))
		return tuple(features)
	
	def extract_feature_values_from_edges(self, edges):
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
	
	def extract_feature_values_from_weighted_edges(self, edges):
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

	def extract_feature_values_from_rules(self, rules):
		ngrams = []
		for rule in rules:
			ngrams.append(list(algorithms.ngrams.generate_n_grams(\
				rule.seq() + ('#',), 1)))
		return (tuple(ngrams),)

