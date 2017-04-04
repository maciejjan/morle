from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.rules import Rule
from utils.files import read_tsv_file

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

    def __eq__(self, other) -> bool:
        return self.to_tuple() == other.to_tuple()

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
            self.edges_list.append(edge)

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
        raise NotImplementedError()

# class Lexicon:
#     def __init__(self, rootdist=None, ruleset=None):
#         self.nodes = {}
#         self.roots = set()
#         self.edges_by_rule = defaultdict(lambda: list())
#         self.cost = 0.0
#         self.rules_c = defaultdict(lambda: 0)
#         self.transducer = None
#     
#     def __len__(self):
#         return len(self.nodes)
#     
#     def __contains__(self, key):
#         return self.nodes.__contains__(key)
#     
#     def keys(self):
#         return self.nodes.keys()
# 
#     def values(self):
#         return self.nodes.values()
#     
#     def __getitem__(self, key):
#         return self.nodes[key]
#     
#     def __setitem__(self, key, val):
#         self.nodes[key] = val
#     
#     def __delitem__(self, key):
#         del self.nodes[key]
#     
#     def iter_nodes(self):
#         return self.nodes.values()
#     
#     def iter_edges(self):
#         for node in self.iter_nodes():
#             for edge in node.edges:
#                 yield edge
# 
#     def recompute_cost(self, model):
#         self.cost = sum(rt.cost for rt in self.roots) +\
#             sum(edge.cost for edge in self.iter_edges()) +\
#             model.null_cost()
#     
#     def recompute_all_costs(self, model):
#         self.cost = 0
#         for node in self.iter_nodes():
#             node.cost = model.word_cost(node)
#             self.cost += node.cost
#         for edge in self.iter_edges():
#             edge.cost = model.edge_cost(edge)
#             self.cost += edge.cost - edge.target.cost
#     
#     def add_node(self, node):
#         key = node.key
#         self.nodes[key] = node
#         self.roots.add(node)
#         self.cost += node.cost
#     
# #     def trigram_hash(self):
# #         trh = TrigramHash()
# #         for node in self.nodes.values():
# #             trh.add(node)
# #         return trh
# 
#     # TODO full cycle detection
#     def check_if_edge_possible(self, edge):
# #        n1, n2 = self.nodes[edge.source], self.nodes[edge.target]
#         if edge.source.parent is not None and edge.source.parent == edge.target:
#             raise Exception('Cycle detected: %s, %s' % (edge.source.key, edge.target.key))
#         if edge.target.parent is not None:
#             raise Exception('%s has already got an ingoing edge.' % edge.target.key)
#     
#     def has_edge(self, edge):
#         return edge in edge.source.edges
# 
#     def add_edge(self, edge):
#         self.check_if_edge_possible(edge)
# #        w1, w2 = self.nodes[edge.source], self.nodes[edge.target]
#         edge.source.edges.append(edge)
#         edge.target.parent = edge.source
#         self.roots.remove(edge.target)
#         self.edges_by_rule[edge.rule].append(edge)
#         self.rules_c[edge.rule] += 1
#         self.cost += edge.cost - edge.target.cost
#     
#     def remove_edge(self, edge):
#         edge.target.parent = None
#         self.roots.add(edge.target)
#         edge.source.edges.remove(edge)
#         self.edges_by_rule[edge.rule].remove(edge)
#         self.rules_c[edge.rule] -= 1
#         self.cost -= edge.cost - edge.target.cost
#     
#     def reset(self):
#         self.edges_by_rule = defaultdict(lambda: list())
#         for node in self.iter_nodes():
#             node.parent = None
#             node.edges = []
#             self.roots.add(node)
#     
# # TODO deprecated
# #     def build_transducer(self, print_progress=False):
# #         self.alphabet =\
# #             tuple(sorted(set(
# #                 itertools.chain(*(n.word+n.tag for n in self.nodes.values()))
# #             )))
# #         self.transducer =\
# #             algorithms.fst.binary_disjunct(
# #                 [algorithms.fst.seq_to_transducer(\
# #                          n.seq(), alphabet=self.alphabet)\
# #                      for n in self.nodes.values()],
# #                 print_progress=print_progress
# #             )
#     
#     def save_to_file(self, filename):
#         def write_subtree(fp, source, target, rule):
#             line = (source.key, target.key, str(rule)) if source is not None\
#                 else ('', target.key, '')
#             if settings.WORD_FREQ_WEIGHT > 0.0:
#                 line = line + (target.freq,)
#             if settings.WORD_VEC_WEIGHT > 0.0:
#                 line = line + (settings.VECTOR_SEP.join(map(str, n2.vec)),)
#             write_line(fp, line)
#             for edge in target.edges:
#                 write_subtree(fp, target, edge.target, edge.rule)
#         with open_to_write(filename) as fp:
#             for rt in self.roots:
#                 write_subtree(fp, None, rt, None)
#     
#     @staticmethod
#     def init_from_wordlist(filename):
#         '''Create a lexicon with no edges from a wordlist.'''
#         lexicon = Lexicon()
#         for node_data in read_tsv_file(filename, get_wordlist_format()):
#             # if the input file contains base words -> ignore them for now
#             if shared.config['General'].getboolean('supervised'):
#                 node_data = node_data[1:]
# #             try:
#             lexicon.add_node(LexiconNode(*node_data))
# #             except Exception:
# #                 logging.getLogger('main').warning('ignoring %s' % node_data[0])
#         return lexicon
# 
#     # init from file of form word_1<TAB>word_2 (extract rules)
#     # TODO: change name
#     @staticmethod
#     def init_from_training_file(filename):
#         '''Create a lexicon from a list of word pairs.'''
#         lexicon = Lexicon()
#         for node_1_key, node_2_key in read_tsv_file(filename, (str, str)):
#             if node_1_key not in lexicon:
#                 lexicon.add_node(LexiconNode(node_1_key))
#             if node_2_key not in lexicon:
#                 lexicon.add_node(LexiconNode(node_2_key))
#             n1, n2 = lexicon[node_1_key], lexicon[node_2_key]
#             rule = algorithms.align.extract_rule(n2, n1)
#             lexicon.add_edge(LexiconEdge(n2, n1, rule))
#         return lexicon
# 
#     @staticmethod
#     def load_from_file(filename):
#         lexicon = Lexicon()
#         for line in read_tsv_file(filename, settings.LEXICON_FORMAT):
#             node_data = (line[1],) + line[3:]
#             lexicon.add_node(LexiconNode(*node_data))
#             if line[0] and line[2]:
#                 lexicon.add_edge(LexiconEdge(*line[:3]))
#         return lexicon
#         # TODO rootdist
# 
#     def save_model(self, rules_file, lexicon_file):
#         '''Save rules and lexicon.'''
#         self.ruleset.save_to_file(rules_file)
#         self.save_to_file(lexicon_file)
# 
#     @staticmethod
#     def load_model(rules_file, lexicon_file):
#         '''Load rules and lexicon.'''
#         ruleset = RuleSet.load_from_file(rules_file)
#         lexicon = Lexicon.load_from_file(lexicon_file, ruleset)
#         return lexicon
    
