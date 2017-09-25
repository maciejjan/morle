from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.rules import Rule, RuleSet
from utils.files import open_to_write, read_tsv_file, write_line
import shared

from collections import defaultdict
import networkx as nx
import numpy as np
import random
from typing import Dict, Iterable, List, Set, Tuple, Union



# TODO rename: GraphEdge -> Edge
class GraphEdge:
    def __init__(self, 
                 source :LexiconEntry,
                 target :LexiconEntry, 
                 rule :Rule,
                 **kwargs) -> None:
        self.source = source
        self.target = target
        self.rule = rule
        self.attr = kwargs

    def key(self) -> Tuple[LexiconEntry, LexiconEntry, Rule]:
        return (self.source, self.target, self.rule)

    def __hash__(self) -> int:
        return self.key().__hash__()

    def __eq__(self, other) -> bool:
        return self.key() == other.key()

    def to_tuple(self) -> Tuple[LexiconEntry, LexiconEntry, Rule, Dict]:
        attr = self.attr
        attr['object'] = self
        return (self.source, self.target, self.rule, attr)


class EdgeSet:
    '''Class responsible for reading/writing a list of edges from/to a file
       and indexing (i.e. assigning IDs to) the edges.'''

    def __init__(self) -> None:
        self.items = []               # type: List[GraphEdge]
        self.index = {}               # type: Dict[GraphEdge, int]
        self.edge_ids_by_rule = {}    # type: Dict[Rule, List[int]]
        self.next_id = 0
        if shared.config['Models'].get('edge_feature_model') != 'none':
            dim = shared.config['Features'].getint('word_vec_dim')
            self.feature_matrix = np.ndarray((0, dim))

    def __iter__(self) -> Iterable[GraphEdge]:
        return iter(self.items)

    def __getitem__(self, idx :int) -> GraphEdge:
        return self.items[idx]

    def __contains__(self, edge :GraphEdge) -> bool:
        return edge in self.index

    def __len__(self) -> int:
        return len(self.items)

    def add(self, edges :Union[GraphEdge, Iterable[GraphEdge]]) -> None:
        if isinstance(edges, GraphEdge):
            edges = [edges]
        if not isinstance(edges, list):
            edges = list(edges)             # because we need two iterations
        for edge in edges:
            self.items.append(edge)
            self.index[edge] = self.next_id
            if edge.rule not in self.edge_ids_by_rule:
                self.edge_ids_by_rule[edge.rule] = []
            self.edge_ids_by_rule[edge.rule].append(self.next_id)
            self.next_id += 1
        if shared.config['Models'].get('edge_feature_model') != 'none':
            for edge in edges:
                edge.attr['vec'] = edge.target.vec-edge.source.vec
            self.feature_matrix = \
                np.vstack((self.feature_matrix,
                           np.array([edge.attr['vec'] for edge in edges])))

    def get_id(self, edge :GraphEdge) -> int:
        return self.index[edge]

    def get_edge_ids_by_rule(self) -> Dict[Rule, List[int]]:
        return self.edge_ids_by_rule

    def save(self, filename :str) -> None:
        with open_to_write(filename) as fp:
            for edge in self.__iter__():
                write_line(fp, edge.to_tuple())

    @staticmethod
    def load(filename :str, lexicon :Lexicon, rule_set :RuleSet) -> 'EdgeSet':
        result = EdgeSet()
        result.add(GraphEdge(lexicon[source], lexicon[target], rule_set[rule])\
                   for source, target, rule in read_tsv_file(filename))
        return result


class Graph(nx.MultiDiGraph):

    def add_edge(self, edge :GraphEdge) -> None:
        super().add_edge(*edge.to_tuple())

    def remove_edge(self, edge :GraphEdge) -> None:
        super().remove_edge(edge.source, edge.target, edge.rule)
        if self.find_edges(edge.source, edge.target):
            raise Exception('remove_edge apparently didn\'t work')

    def edges_iter(self) -> Iterable[GraphEdge]:
        return (attr['object'] for source, target, rule, attr in \
                                   super().edges_iter(keys=True, data=True))

    def find_edges(self, source :LexiconEntry, target :LexiconEntry) \
                  -> List[GraphEdge]:
        result = []
        if source in self and target in self[source]:
            for rule, attr in self[source][target].items():
                result.append(GraphEdge(source, target, rule, **attr))
        return result


class Branching(Graph):
    # TODO well-formedness conditions etc.
    def __init__(self):
        super().__init__()
        self.edges_by_rule = defaultdict(lambda: list())

    def add_edge(self, edge :GraphEdge) -> None:
        super().add_edge(edge)
        self.edges_by_rule[edge.rule].append(edge)

    def remove_edge(self, edge :GraphEdge) -> None:
        super().remove_edge(edge)
        self.edges_by_rule[edge.rule].remove(edge)

    def is_edge_possible(self, edge :GraphEdge) -> bool:
        if edge.source == edge.target:
            return False
        if self.parent(edge.target) is not None:
            return False
        if self.has_path(edge.target, edge.source):
            return False
        return True

    def parent(self, node :LexiconEntry) -> LexiconEntry:
        predecessors = self.predecessors(node)
        return predecessors[0] if predecessors else None

    def root(self, node :LexiconEntry) -> LexiconEntry:
        if self.parent(node) is None:
            return None
        root = self.parent(node)
        while root.parent() is not None:
            root = self.parent(root)
        return root

    def has_path(self, source :LexiconEntry, target :LexiconEntry) -> bool:
        node = target
        seen_nodes = set()
        while node is not None:
            if node == source:
                return True
            seen_nodes.add(node)
            node = self.parent(node)
            if node in seen_nodes:
                raise Exception('Cycle detected!')
        return False

    def get_edges_for_rule(self, rule :Rule) -> List[GraphEdge]:
        return self.edges_by_rule[rule]


class FullGraph(Graph):
    # TODO immutable, loaded from file
    def __init__(self, lexicon :Lexicon, edge_set :EdgeSet) -> None:
        super().__init__()
#         self.edges_list = []
        self.lexicon = lexicon
        self.edge_set = edge_set
#         self.rules = {}       # type: Dict[str, Rule]
        for lexentry in lexicon:
            self.add_node(lexentry)
        for edge in edge_set:
            self.add_edge(edge)

    def add_edge(self, edge :GraphEdge) -> None:
        super().add_edge(edge)

#     def load_edges_from_file(self, filename :str) -> None:
#         starting_id = len(self.edges_list) + 1
#         rules = {}              # type: Dict[str, Rule]
#         for cur_id, (w1, w2, rule_str) in enumerate(read_tsv_file(filename),\
#                                                     starting_id):
#             v1, v2 = self.lexicon[w1], self.lexicon[w2]
#             if not rule_str in rules:
#                 rules[rule_str] = Rule.from_string(rule_str)
#             rule = rules[rule_str]
#             edge = GraphEdge(v1, v2, rule, id=cur_id)
#             self.add_edge(edge)

#     def iter_edges(self) -> Iterable[GraphEdge]:
#         return iter(self.edges_list)

    def random_edge(self) -> GraphEdge:
        # choose an edge with uniform probability
        return random.choice(self.edge_set.items)

    def empty_branching(self) -> Branching:
        branching = Branching()
        branching.add_nodes_from(self)
        return branching

    def random_branching(self) -> Branching:
        # choose some edges randomly and compose a branching out of them
        edges = list(iter(self.edge_set))
        random.shuffle(edges)
        branching = self.empty_branching()
        for edge in edges:
            if branching.is_edge_possible(edge) and random.random() < 0.5:
                branching.add_edge(edge)
        return branching
#         edge_ids = list(range(len(self.edge_set.items)))
#         random.shuffle(edge_indices)
#         edge_indices = edge_indices[:random.randrange(len(edge_indices))]
# 
#         branching = self.empty_branching()
#         for idx in edge_indices:
#             edge = self.edge_set.[idx]
#             if branching.is_edge_possible(edge):
#                 branching.add_edge(edge)
#         return branching

    def optimal_branching(self, model :'PointModel') -> Branching:
        graph = nx.DiGraph()
        for edge in self.edge_set:
            weight = model.root_cost(edge.target) - model.edge_cost(edge)
            if graph.has_edge(edge.source, edge.target):
                if weight > graph[edge.source][edge.target]['weight']:
                    graph[edge.source][edge.target]['weight'] = weight
            else:
                graph.add_edge(edge.source, edge.target, weight=weight)

        branching = nx.maximum_branching(graph)

        result = Branching()
        for source, target in branching.edges_iter():
            result.add_edge(min(self.find_edges(source, target), \
                                key=lambda e: model.root_cost(e.target) -\
                                              model.edge_cost(e)))
        return result

    def restriction_to_ruleset(self, ruleset :Set[Rule]) -> 'FullGraph':
        # TODO
        result = FullGraph(self.lexicon)
        for edge in self.edges_list:
            if edge.rule in ruleset:
                result.add_edge(edge)
        return result

    # TODO deprecated -- replaced with EdgeSet.save() and Lexicon.save()
#     def save_to_file(self, filename :str) -> None:
#         with open_to_write(filename) as fp:
#             for source, target, rule in self.edges_iter(keys=True):
#                 write_line(fp, (source, target, rule))

