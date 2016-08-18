#from datastruct.lexicon import *
#from datastruct.rules import Rule
#from models.features.extractor import FeatureValueExtractor
#from models.features.point import PointFeatureSet
from models.generic import Model
#from utils.files import *
#
#from numpy.linalg import norm

class PointModel(Model):

    def __init__(self, lexicon=None, rules=None, rule_domsizes=None):
        Model.__init__(self, lexicon, rules, rule_domsizes)
        self.model_type = 'point'

    def fit_to_sample(self, sample):
        raise NotImplementedError()


#    def __init__(self, lexicon=None, rules=None, edges=None):
#        self.extractor = FeatureValueExtractor()
#        self.word_prior = None
#        self.rule_prior = None
#        self.rule_features = {}
#        self.cost = 0.0
#    
#    def add_rule(self, rule, rule_features):
#        self.rule_features[rule] = rule_features
#        self.cost += self.rule_cost(rule)
#    
#    def delete_rule(self, rule):
#        self.cost -= self.rule_cost(rule)
#        del self.rule_features[rule]
#    
#    def num_rules(self):
#        return len(self.rule_features)
#    
#    def has_rule(self, rule):
#        if isinstance(rule, Rule):
#            return rule in self.rule_features
#        elif isinstance(rule, str):
#            return Rule.from_string(rule) in self.rule_features
#
#    #TODO generator -- fitting only the first feature?!
#    def fit_rootdist(self, lexicon):
#        self.word_prior = PointFeatureSet.new_root_feature_set()
#        self.word_prior.fit(
#            self.extractor.extract_feature_values_from_nodes(lexicon.values()))
#    
#    def fit_ruledist(self, rule_features=None):
#        if rule_features is None:
#            rule_features = self.rule_features
#        self.rule_prior = PointFeatureSet.new_rule_feature_set()
#        self.rule_prior.fit(
#            self.extractor.extract_feature_values_from_rules(
#                (rule, features)\
#                    for rule, features in rule_features.items()))
#    
#    def fit_rule(self, rule, edges, domsize):
#        features = PointFeatureSet.new_edge_feature_set(domsize)
#        features.fit(self.extractor.extract_feature_values_from_edges(edges))
#        return features
#    
#    def weighted_fit_rule(self, rule, edges, domsize):
#        features = PointFeatureSet.new_edge_feature_set(domsize)
#        features.weighted_fit(\
#            self.extractor.extract_feature_values_from_weighted_edges(edges))
#        return features
#    
#    def fit_to_lexicon(self, lexicon):
#        for rule, edges in lexicon.edges_by_rule.items():
#            domsize = self.rule_features[rule][0].trials
#            features = self.fit_rule(rule, edges, domsize)
#            old_cost = self.cost
#            self.delete_rule(rule)
#            # delete the rule if it brings more loss than gain -- TODO to a separate function
#            rule_gain = features.cost(\
#                self.extractor.extract_feature_values_from_edges(edges)) -\
#                self.rule_cost(rule) -\
#                features.null_cost() +\
#                sum(e.target.cost for e in edges)
#            if rule_gain <= 0.0:
#                pass
#            else:
#                self.add_rule(rule, features)
#        # delete the rules with no edges
#        rules_to_delete = [r for r in self.rule_features if r not in lexicon.edges_by_rule]
#        for r in rules_to_delete:
#            self.delete_rule(r)
#
#    def fit_to_sample(self, edges_by_rule):
#        for rule, edges in edges_by_rule.items():
#            domsize = self.rule_features[rule][0].trials
#            features = self.weighted_fit_rule(rule, edges, domsize)
#            self.delete_rule(rule)
#            # delete the rule if it brings more loss than gain (TODO weighted)
#            rule_gain = features.weighted_cost(\
#                self.extractor.extract_feature_values_from_weighted_edges(edges)) -\
#                self.rule_cost(rule) -\
#                features.null_cost() +\
#                sum(e.target.cost*weight for e, weight in edges)
#            if rule_gain <= 0.0:
##                print('Deleting rule: %s gain=%f' % (str(rule), rule_gain))
##                self.delete_rule(rule)
#                pass
#            else:
#                self.add_rule(rule, features)
#
#    def rule_split_gain(self, rule_1, edges_1, domsize_1,\
#            rule_2, edges_2, domsize_2):
#        edges_3 = edges_1 + edges_2
#        features_1 = PointFeatureSet.new_edge_feature_set(domsize_1)
#        features_2 = PointFeatureSet.new_edge_feature_set(domsize_2)
#        features_3 = PointFeatureSet.new_edge_feature_set(domsize_2)
#        values_1 = self.extractor.extract_feature_values_from_edges(edges_1)
#        values_2 = self.extractor.extract_feature_values_from_edges(edges_2)
#        values_3 = self.extractor.extract_feature_values_from_edges(edges_3)
#        features_1.fit(values_1)
#        features_2.fit(values_2)
#        features_3.fit(values_3)
#        return features_3.cost(values_3) - features_1.cost(values_1) -\
#            features_2.cost(values_2) - self.rule_cost(rule_1) +\
#            (self.rule_cost(rule_2) if len(edges_2) == 0 else 0) +\
#            features_3.null_cost() - features_1.null_cost() -\
#            features_2.null_cost()
#    
#    def null_cost(self):
#        return sum(features.null_cost()\
#            for features in self.rule_features.values())
#
#    def node_cost(self, node):
#        return self.word_prior.cost(\
#            self.extractor.extract_feature_values_from_nodes((node,)))
#    
#    # TODO cost of a single edge is not computed properly!
#    # -> binomial cost of all edges is computed
#    # how to compute the cost of a single edge? (sometimes Bernoulli, sometimes binomial)
#    def edge_cost(self, edge):
#        return self.rule_features[edge.rule].cost(\
#            self.extractor.extract_feature_values_from_edges((edge,)))
#
#    def weighted_edge_cost(self, edge, weight):
#        return self.rule_features[edge.rule].weighted_cost(\
#            self.extractor.extract_feature_values_from_weighted_edges((edge, weight)))
#    
#    def rule_cost(self, rule, rule_features=None):
#        if rule_features is None:
#            if rule in self.rule_features:
#                rule_features = self.rule_features[rule]
#            else:
#                rule_features = ()
#        return self.rule_prior.cost(\
#            self.extractor.extract_feature_values_from_rules(
#                ((rule, rule_features),)))
#    
#    def save_to_file(self, filename):
#        # forget the transducers, because they are not serializable
#        for rule in self.rule_features:
#            rule.transducer = None
#        Model.save_to_file(self, filename)
#    
#    def save_rule_stats(self, filename):
#
#        def feature_stats(feature):
#            if isinstance(feature, PointBinomialFeature):
#                return (feature.trials, feature.prob, feature.trials*feature.prob)
#            elif isinstance(feature, PointGaussianGammaGammaFeature):
#                return (feature.mean, feature.var)
#            elif isinstance(feature, PointGaussianGaussianGammaFeature):
#                return (norm(feature.mean), norm(feature.var))
#            elif isinstance(feature, PointGaussianInverseChiSquaredFeature):
#                if feature.dim == 1:
#                    return (feature.mean, feature.var)
#                else:
#                    return (norm(feature.mean), norm(feature.var))
#
#        with open_to_write(filename) as fp:
#            for rule, features in self.rule_features.items():
#                write_line(
#                    fp,
#                    (rule,) + sum(map(feature_stats, features), ())
#                )
#
##def fit_model_to_graph(lexicon, trh, graph_file):
#def fit_model_to_graph(lexicon, graph_file):
#    model = PointModel()
#    model.fit_word_prior(lexicon)
#    rule_features = {}
#    for rule_str, wordpairs in read_tsv_file_by_key(graph_file, key=3,\
#            print_progress=True):
#        rule = Rule.from_string(rule_str)
#        edges = [LexiconEdge(lexicon[w1], lexicon[w2], rule)\
#            for w1, w2 in wordpairs]
##        domsize = rule.compute_domsize(trh)
#        domsize = rule.compute_domsize(lexicon)
#        rule_features[rule] = model.fit_rule(rule, edges, domsize)
##        model.add_rule(rule, rule_featuers[rule])
#    model.fit_rule_prior(rule_features)
#    for rule, features in rule_features.items():
#        model.add_rule(rule, features)
#    return model
