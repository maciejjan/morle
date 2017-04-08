from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.rules import Rule
from utils.files import open_to_write, read_tsv_file, write_line

from collections import defaultdict
import networkx as nx
import random
from typing import Dict, Iterable, List, Set, Tuple



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
        return (self.source, self.target, self.rule, self.attr)


class Graph(nx.MultiDiGraph):

    def add_edge(self, edge :GraphEdge) -> None:
        super().add_edge(*edge.to_tuple())

    def remove_edge(self, edge :GraphEdge) -> None:
        super().remove_edge(edge.source, edge.target, edge.rule)
        if self.find_edges(edge.source, edge.target):
            raise Exception('remove_edge apparently didn\'t work')

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
    def __init__(self, lexicon :Lexicon = None) -> None:

        assert lexicon is not None
        super().__init__()
        self.edges_list = []
        self.lexicon = lexicon
#         self.rules = {}       # type: Dict[str, Rule]
        for entry in lexicon.entries():
            self.add_node(entry)

    def add_edge(self, edge :GraphEdge) -> None:
        super().add_edge(edge)
        self.edges_list.append(edge)

    def load_edges_from_file(self, filename :str) -> None:
        starting_id = len(self.edges_list) + 1
        rules = {}              # type: Dict[str, Rule]
        for cur_id, (w1, w2, rule_str) in enumerate(read_tsv_file(filename),\
                                                    starting_id):
            v1, v2 = self.lexicon[w1], self.lexicon[w2]
            if not rule_str in rules:
                rules[rule_str] = Rule.from_string(rule_str)
            rule = rules[rule_str]
            edge = GraphEdge(v1, v2, rule, id=cur_id)
            self.add_edge(edge)

    def iter_edges(self) -> Iterable[GraphEdge]:
        return iter(self.edges_list)

    def random_edge(self) -> GraphEdge:
        # choose an edge with uniform probability
        return random.choice(self.edges_list)

    def empty_branching(self) -> Branching:
        branching = Branching()
        branching.add_nodes_from(self)
        return branching

    def random_branching(self) -> Branching:
        # choose some edges randomly and compose a branching out of them
        edge_indices = list(range(len(self.edges_list)))
        random.shuffle(edge_indices)
        edge_indices = edge_indices[:random.randrange(len(edge_indices))]

        branching = self.empty_branching()
        for idx in edge_indices:
            edge = self.edges_list[idx]
            if branching.is_edge_possible(edge):
                branching.add_edge(edge)
        return branching

    def restriction_to_ruleset(self, ruleset :Set[Rule]) -> 'FullGraph':
        result = FullGraph(self.lexicon)
        for edge in self.edges_list:
            if edge.rule in ruleset:
                result.add_edge(edge)
        return result

    def save_to_file(self, filename :str) -> None:
        with open_to_write(filename) as fp:
            for source, target, rule in self.edges_iter(keys=True):
                write_line(fp, (source, target, rule))

