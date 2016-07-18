from models.features.generic import StringFeature
from models.features.extractor import FeatureValueExtractor
from models.features.marginal import *
from models.generic import Model

from operator import itemgetter

class MarginalModel(Model):

    def __init__(self, lexicon, ruleset):
        self.extractor = FeatureValueExtractor()
        self.rootdist = None
        self.ruledist = None
        self.rule_features = {}
        self.root_features = None
        self.roots_cost = 0.0
        self.edges_cost = 0.0
        self.rules_cost = 0.0

        self.fit_rootdist(lexicon)
        self.fit_ruledist(ruleset)
        for rule in ruleset:
            self.add_rule(rule, rule.compute_domsize(lexicon))
        self.fit_to_lexicon(lexicon)
    
    def fit_rootdist(self, lexicon):
        self.rootdist = MarginalFeatureSet.new_root_feature_set()
        # fit only the first feature (the MarginalStringFeature)
        self.rootdist[0].fit(\
            self.extractor.extract_feature_values_from_nodes(\
                list(lexicon.iter_nodes()))[0])
    
    def fit_ruledist(self, ruleset):
        self.ruledist = MarginalFeatureSet.new_rule_feature_set()
        # fit only the first feature (the MarginalStringFeature)
        self.ruledist[0].fit(\
            self.extractor.extract_feature_values_from_rules(ruleset)[0])
    
    def fit_to_lexicon(self, lexicon):
        roots_to_add = self.extractor.extract_feature_values_from_nodes(
            lexicon.roots)
        self.rootdist.apply_change(roots_to_add, [])
        for rule, edges in lexicon.edges_by_rule.items():
            edges_to_add = self.extractor.extract_feature_values_from_edges(
                edges)
            self.rule_features[rule].apply_change(edges_to_add, [])

    def reset(self):
        self.rootdist.reset()
        self.ruledist.reset()
        for features in self.rule_features.values():
            features.reset()

    def num_rules(self):
        return len(self.rule_features)
    
    def cost(self):
        return self.rootdist.cost() + self.ruledist.cost() +\
            sum(f.cost() for f in self.rule_features.values())
    
    def add_rule(self, rule, domsize):
        self.rule_features[rule] =\
            MarginalFeatureSet.new_edge_feature_set(domsize)
    
    def cost_of_change(self, edges_to_add, edges_to_remove):
        result = 0.0
        root_changes, changes_by_rule =\
            self.extract_feature_values_for_change(
                edges_to_add, edges_to_remove)
        # apply the changes to roots
        roots_to_add, roots_to_remove = root_changes
        result += self.rootdist.cost_of_change(
            roots_to_add, roots_to_remove)
        # apply the changes to rule features
        for rule, (values_to_add, values_to_remove) in changes_by_rule.items():
            result += self.rule_features[rule].cost_of_change(
                values_to_add, values_to_remove)
        return result

    def apply_change(self, edges_to_add, edges_to_remove):
        root_changes, changes_by_rule =\
            self.extract_feature_values_for_change(
                edges_to_add, edges_to_remove)
        # apply the changes to roots
        roots_to_add, roots_to_remove = root_changes
        self.rootdist.apply_change(roots_to_add, roots_to_remove)
        # apply the changes to rule features
        for rule, (values_to_add, values_to_remove) in changes_by_rule.items():
            self.rule_features[rule].apply_change(
                values_to_add, values_to_remove)

    def extract_feature_values_for_change(self, edges_to_add, edges_to_remove):
        # changes to roots
        roots_to_add = self.extractor.extract_feature_values_from_nodes(
            [e.target for e in edges_to_remove])
        roots_to_remove = self.extractor.extract_feature_values_from_nodes(
            [e.target for e in edges_to_add])
        self.rootdist.apply_change(roots_to_add, roots_to_remove)
        # changes to rule features
        changes_by_rule = defaultdict(lambda: (list(), list()))
        for e in edges_to_add:
            changes_by_rule[e.rule][0].append(
                self.extractor.extract_feature_values_from_edges([e]))
        for e in edges_to_remove:
            changes_by_rule[e.rule][1].append(
                self.extractor.extract_feature_values_from_edges([e]))
        for rule, (edges_to_add_for_rule, edges_to_remove_for_rule) in\
                changes_by_rule.items():
            self.rule_features[rule].apply_change(
                edges_to_add_for_rule, edges_to_remove_for_rule)
        return (roots_to_add, roots_to_remove), changes_by_rule

