from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.rules import Rule, RuleSet
from utils.files import open_to_write, read_tsv_file, write_line
import shared

from collections import defaultdict
import logging
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

    def __str__(self) -> str:
        return '{} -> {} by {}'.format(self.source, self.target, self.rule)

    def to_tuple(self) -> Tuple[LexiconEntry, LexiconEntry, Rule, Dict]:
        attr = self.attr
        attr['object'] = self
        return (self.source, self.target, self.rule, attr)


class EdgeSet:
    '''Class responsible for reading/writing a list of edges from/to a file
       and indexing (i.e. assigning IDs to) the edges.'''

    def __init__(self, lexicon :Lexicon,
                 edges :Union[GraphEdge, Iterable[GraphEdge]] = None) -> None:
        self.lexicon = lexicon
        self.items = []               # type: List[GraphEdge]
        self.index = {}               # type: Dict[GraphEdge, int]
        self.edge_ids_by_rule = {}    # type: Dict[Rule, List[int]]
        self.next_id = 0
        if edges is not None:
            self.add(edges)

    def __iter__(self) -> Iterable[GraphEdge]:
        return iter(self.items)

    def __getitem__(self, idx :int) -> GraphEdge:
        return self.items[idx]

    def __contains__(self, edge :GraphEdge) -> bool:
        return edge in self.index

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return True if self.items else False

    def add(self, edges :Union[GraphEdge, Iterable[GraphEdge]]) -> None:
        if isinstance(edges, GraphEdge):
            edges = [edges]
        for edge in edges:
            self.items.append(edge)
            self.index[edge] = self.next_id
            if edge.rule not in self.edge_ids_by_rule:
                self.edge_ids_by_rule[edge.rule] = []
            self.edge_ids_by_rule[edge.rule].append(self.next_id)
            self.next_id += 1

    def remove(self, edges :Union[GraphEdge, Iterable[GraphEdge]]) -> None:
        logging.getLogger('main').debug('Number of edges before deletion: {}'\
                                        .format(len(self.items)))
        if isinstance(edges, GraphEdge):
            edges = [edges]
        edges_to_remove_set = set(edges)
        self.items = [edge for edge in self.items \
                      if edge not in edges_to_remove_set]
        self.index = {}
        self.edge_ids_by_rule = {}
        for i, edge in enumerate(self.items):
            self.index[edge] = i
            if edge.rule not in self.edge_ids_by_rule:
                self.edge_ids_by_rule[edge.rule] = []
            self.edge_ids_by_rule[edge.rule].append(i)
        self.next_id = len(self.items)
        logging.getLogger('main').debug('Number of edges after deletion: {}'\
                                        .format(len(self.items)))

    def get_id(self, edge :GraphEdge) -> int:
        return self.index[edge]

    def get_edge_ids_by_rule(self) -> Dict[Rule, List[int]]:
        return self.edge_ids_by_rule

    def save(self, filename :str) -> None:
        with open_to_write(filename) as fp:
            for edge in self.__iter__():
                write_line(fp, edge.to_tuple()[:3])

    @staticmethod
    def load(filename :str, lexicon :Lexicon, rule_set :RuleSet) -> 'EdgeSet':
        result = EdgeSet(lexicon)
        edge_iter = (GraphEdge(lexicon[source],
                               lexicon[target],
                               rule_set[rule]) \
                     for source, target, rule in read_tsv_file(filename))
        return EdgeSet(lexicon, edge_iter)


class Graph(nx.MultiDiGraph):
    def __init__(self) -> None:
        super().__init__()
        self.edges_by_source = defaultdict(lambda: set())
        self.edges_by_target = defaultdict(lambda: set())

    def add_edge(self, edge :GraphEdge) -> None:
        super().add_edge(*edge.to_tuple())
        self.edges_by_source[edge.source].add(edge)
        self.edges_by_target[edge.target].add(edge)

    def remove_edge(self, edge :GraphEdge) -> None:
        super().remove_edge(edge.source, edge.target, edge.rule)
        self.edges_by_source[edge.source].remove(edge)
        self.edges_by_target[edge.target].remove(edge)

    def edges_iter(self) -> Iterable[GraphEdge]:
        return (attr['object'] for source, target, rule, attr in \
                                   super().edges_iter(keys=True, data=True))

    def edges_between(self, source :LexiconEntry, target :LexiconEntry) \
                     -> Set[GraphEdge]:
        return list(self.edges_by_source[source] & self.edges_by_target[target])
#         result = []
#         if source in self and target in self[source]:
#             for rule, attr in self[source][target].items():
#                 result.append(GraphEdge(source, target, rule, **attr))
#         return result

    def ingoing_edges(self, target :LexiconEntry) -> List[GraphEdge]:
        return list(self.edges_by_target[target])

    def outgoing_edges(self, source :LexiconEntry) -> List[GraphEdge]:
        return list(self.edges_by_source[source])


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
        if len(predecessors) > 1:
            raise Exception('More than one predecessor: {}'.format(node))
        return predecessors[0] if predecessors else None

    def root(self, node :LexiconEntry) -> LexiconEntry:
        root = node
        while self.parent(root) is not None:
            root = self.parent(root)
        return root

    def depth(self, node :LexiconEntry) -> int:
        if self.parent(node) is None:
            return 1
        else:
            return 1 + self.depth(self.parent(node))

    def subtree_size(self, node :LexiconEntry) -> int:
        return 1 + \
               sum(self.subtree_size(child) for child in self.successors(node))

    def count_nonleaves(self, node :LexiconEntry) -> int:
        if not self.successors(node):
            return 0
        else:
            return 1 + sum(self.count_nonleaves(child) \
                           for child in self.successors(node))

    def height(self, node :LexiconEntry) -> int:
        child_heights = [self.height(child) for child in self.successors(node)]
        if not child_heights:
            return 1
        else:
            return max(child_heights) + 1

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

    def path(self, source :LexiconEntry, target :LexiconEntry) \
            -> List[LexiconEntry]:
        if source == target:
            return [source]
        else:
            return self.path(source, self.parent(target)) + [target]

    def get_edges_for_rule(self, rule :Rule) -> List[GraphEdge]:
        return self.edges_by_rule[rule]


class FullGraph(Graph):
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

    def remove_edges(self, edges :List[GraphEdge]) -> None:
        self.edge_set.remove(edges)
        for edge in edges:
            super().remove_edge(edge)

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
        logging.getLogger('main').debug(\
            'random_branching(): {} potential edges'.format(len(edges)))
        random.shuffle(edges)
        branching = self.empty_branching()
        for edge in edges:
            if branching.is_edge_possible(edge) and random.random() < 0.5:
                branching.add_edge(edge)
        return branching

    # TODO edge weighting: use matrix operations!!!
    def optimal_branching(self, model :'ModelSuite') -> Branching:
        root_costs = model.roots_cost(self.lexicon)
        edge_costs = model.edges_cost(self.edge_set)
        graph = nx.DiGraph()
        for edge in self.edge_set:
            # weight is the negative cost, i.e. how much is "saved"
            # by including this edge (because we look for maximum branching)
            weight = root_costs[self.lexicon.get_id(edge.target)] -\
                     edge_costs[self.edge_set.get_id(edge)]
            if graph.has_edge(edge.source, edge.target):
                if weight > graph[edge.source][edge.target]['weight']:
                    graph[edge.source][edge.target]['weight'] = weight
            else:
                graph.add_edge(edge.source, edge.target, weight=weight)

        branching = nx.maximum_branching(graph)

        result = Branching()
        for source, target in branching.edges_iter():
            result.add_edge(\
                max(self.find_edges(source, target), \
                    key=lambda e: root_costs[self.lexicon.get_id(e.target)] -\
                                  edge_costs[self.edge_set.get_id(e)]))
        return result

    def restriction_to_ruleset(self, ruleset :Set[Rule]) -> 'FullGraph':
        # TODO
        result = FullGraph(self.lexicon)
        for edge in self.edges_list:
            if edge.rule in ruleset:
                result.add_edge(edge)
        return result

    def remove_isolated_nodes(self) -> None:
        '''Remove nodes that are not part of any edge'''
        # FIXME a very dirty implementation!
        new_lexicon = Lexicon()
        new_lexicon.add(entry for entry in self.lexicon \
                        if self.ingoing_edges(entry) +\
                           self.outgoing_edges(entry))
        self.lexicon = new_lexicon

