from datastruct.graph import GraphEdge, Branching
from datastruct.lexicon import LexiconEntry
from datastruct.rules import Rule
from models.features.extractor import FeatureValueExtractor
from models.features.factory import FeatureSetFactory
from models.features.generic import FeatureSet
from utils.files import read_tsv_file, write_tsv_file

from collections import defaultdict
from typing import Iterable, List


class Model:
    '''Keeps track of sufficient statistics necessary to determine the
       likelihood of model components: V, E and R.'''

    def __init__(self) -> None:
        if not hasattr(self, 'model_type'):
            self.model_type = 'generic'
        self.extractor = FeatureValueExtractor()
        self.rule_features = {}     # type: Dict[Rule, FeatureSet]
        self.rootdist = FeatureSetFactory.new_root_feature_set(self.model_type)
        self.ruledist = FeatureSetFactory.new_rule_feature_set(self.model_type)
        self.roots_cost = 0.0
        self.edges_cost = 0.0
        self.rules_cost = 0.0

    def cost(self):
        return self.roots_cost + self.rules_cost + self.edges_cost

    def iter_rules(self) -> Iterable[Rule]:
        return iter(self.rule_features.keys())

    def num_rules(self) -> int:
        return len(self.rule_features)

    def reset(self) -> None:
        '''Remove any information about the graph (roots or edges).'''
        for f in self.rule_features.values():
            f.reset()
        self.roots_cost = 0.0
        self.edges_cost = sum(f.null_cost()\
                              for f in self.rule_features.values())

    def fit_rootdist(self, roots :Iterable[LexiconEntry]) -> None:
        self.rootdist[0].fit(\
            self.extractor.extract_feature_values_from_nodes(roots)[0])

    def fit_ruledist(self, rules :Iterable[Rule]) -> None:
        self.ruledist[0].fit(\
            self.extractor.extract_feature_values_from_rules(rules)[0])

    def add_rule(self, rule :Rule, domsize :int) -> None:
        self.rule_features[rule] =\
            FeatureSetFactory.new_edge_feature_set(self.model_type, domsize)
        self.rules_cost += self.rule_cost(rule)
        self.edges_cost += self.rule_features[rule].null_cost()

    def remove_rule(self, rule :Rule) -> None:
        self.edges_cost -= self.rule_features[rule].cost()
        self.rules_cost -= self.rule_cost(rule)
        del self.rule_features[rule]

    def root_cost(self, root :LexiconEntry) -> None:
        return self.rootdist.cost_of_change(\
            self.extractor.extract_feature_values_from_nodes([root]), [])

    def rule_cost(self, rule :Rule) -> float:
        return self.ruledist.cost_of_change(\
            self.extractor.extract_feature_values_from_rules([rule]), [])

    # TODO rename!!! - no fitting takes place here
    def fit_to_branching(self, branching :Branching) -> None:
        self.reset()
        # add roots
        roots = [node for node in branching.nodes_iter() \
                      if not branching.predecessors(node)]
        roots_feat = self.extractor.extract_feature_values_from_nodes(roots)
        self.roots_cost = self.rootdist.cost_of_change(roots_feat, [])
        self.rootdist.apply_change(roots_feat, [])
        # add edges
        edges_by_rule = {}
        for source, target, rule, attr in \
                                  branching.edges_iter(keys=True, data=True):
            edge = GraphEdge(source, target, rule, **attr)
            if rule not in edges_by_rule:
                edges_by_rule[rule] = []
            edges_by_rule[rule].append(edge)
        for rule, edges in edges_by_rule.items():
            edges_feat = \
                self.extractor.extract_feature_values_from_edges(edges)
            self.edges_cost += \
                self.rule_features[rule].cost_of_change(edges_feat, [])
            self.rule_features[rule].apply_change(edges_feat, [])

    def cost_of_change(self, edges_to_add :List[GraphEdge], 
                             edges_to_remove :List[GraphEdge]) -> float:
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
        self.roots_cost += \
            self.rootdist.cost_of_change(roots_to_add, roots_to_remove)
        self.rootdist.apply_change(roots_to_add, roots_to_remove)
        # apply the changes to rule features
        for rule, (values_to_add, values_to_remove) in changes_by_rule.items():
            self.edges_cost += self.rule_features[rule].cost_of_change(
                values_to_add, values_to_remove)
            self.rule_features[rule].apply_change(
                values_to_add, values_to_remove)

    def extract_feature_values_for_change(self, edges_to_add, edges_to_remove):
        # changes to roots
        roots_to_add = self.extractor.extract_feature_values_from_nodes(
            [e.target for e in edges_to_remove])
        roots_to_remove = self.extractor.extract_feature_values_from_nodes(
            [e.target for e in edges_to_add])
        # changes to rule features
        edges_to_add_by_rule = defaultdict(lambda: list())
        edges_to_remove_by_rule = defaultdict(lambda: list())
        rules = set()
        for e in edges_to_add:
            edges_to_add_by_rule[e.rule].append(e)
            rules.add(e.rule)
        for e in edges_to_remove:
            edges_to_remove_by_rule[e.rule].append(e)
            rules.add(e.rule)
        changes_by_rule = {}
        for rule in rules:
            changes_by_rule[rule] = (\
                self.extractor.extract_feature_values_from_edges(\
                    edges_to_add_by_rule[rule]),
                self.extractor.extract_feature_values_from_edges(\
                    edges_to_remove_by_rule[rule]))
        return (roots_to_add, roots_to_remove), changes_by_rule

    def save_rules_to_file(self, filename :str) -> None:
        rows = ((str(rule), features[0].trials) \
                for rule, features in self.rule_features.items())
        write_tsv_file(filename, rows)

