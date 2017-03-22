# import algorithms.ngrams #TODO deprecated
import shared

class FeatureValueExtractor:
    def __init__(self):
        # cache frequently used configuration items for better performance
        self.rootdist_n = shared.config['Features'].getint('rootdist_n')
        self.use_word_freq = \
            shared.config['Features'].getfloat('word_freq_weight') > 0.0
        self.use_word_vec = \
            shared.config['Features'].getfloat('word_vec_weight') > 0.0

    def extract_feature_values_from_nodes(self, nodes):
        features = []
#         features.append(list(algorithms.ngrams.generate_n_grams(\
#             node.word + node.tag + ('#',), self.rootdist_n)\
#                 for node in nodes))
        features.append([node.symstr for node in nodes])
        if self.use_word_freq:
            features.append(list(node.logfreq for node in nodes))
        if self.use_word_vec:
            features.append(list(node.vec for node in nodes))
        return tuple(features)
    
    def extract_feature_values_from_edges(self, edges):
#         features = [list(1 for e in edges)]
        features = [[1] * len(edges)]
        if self.use_word_freq:
            # source-target, because target-source typically negative
            features.append(\
                [e.source.logfreq - e.target.logfreq for e in edges]
            )
        if self.use_word_vec:
            features.append(\
                list(e.target.vec - e.source.vec for e in edges)
            )
        return tuple(features)
    
    def extract_feature_values_from_weighted_edges(self, edges):
        features = [list((1, w) for e, w in edges)]
        if self.use_word_freq:
            features.append(\
                [(e.source.logfreq - e.target.logfreq, w) for e, w in edges]
            )
        if self.use_word_vec:
            features.append(\
                list((e.target.vec - e.source.vec, w) for e, w in edges)
            )
        return tuple(features)

    def extract_feature_values_from_rules(self, rules):
        seqs = []
        for rule in rules:
            seqs.append(rule.seq())
        return (seqs,)

