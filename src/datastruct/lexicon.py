import algorithms.fst
# from algorithms.ngrams import TrigramHash
# from models.marginal import MarginalModel
from utils.files import *
import shared

from collections import defaultdict
import itertools
import re
import math
import logging

def get_wordlist_format():
    result = [str]
    if shared.config['General'].getboolean('supervised'):
        result.append(str)
    if shared.config['Features'].getfloat('word_freq_weight') > 0:
        result.append(int)
    if shared.config['Features'].getfloat('word_vec_weight') > 0:
        result.append(\
            lambda x: np.array(list(map(float, x.split(shared.format['vector_sep'])))))
    return tuple(result)

# def normalize_word(string):
#     # deal with capital letters
#     if string.isupper():
#         string = '{ALLCAPS}' + string.lower()
#     else:
#         new_string_chars = []
#         for c in string:
#             new_string_chars.append('{CAP}' + c.lower() if c.isupper() else c)
#         string = ''.join(new_string_chars)
#     # perform substritutions
#     for subst_from, subst_to in shared.normalization_substitutions:
#         string = string.replace(subst_from, subst_to)
#     return string
# 
# def unnormalize_word(string):
#     # perform substritutions
#     for subst_from, subst_to in shared.normalization_substitutions:
#         string = string.replace(subst_to, subst_from)
#     # deal with capital letters
#     if string.startswith('{ALLCAPS}'):
#         return string[9:].upper()
#     else:
#         new_string_chars = []
#         while string:
#             if string.startswith('{CAP}'):
#                 new_string_chars.append(string[5].upper())
#                 string = string[6:]
#             else:
#                 new_string_chars.append(string[0])
#                 string = string[1:]
#         return ''.join(new_string_chars)

def normalize_seq(seq):
    result = []
    allcaps = True
    for c in seq:
        if c.isupper() and c not in shared.multichar_symbols:
            result.append(c.lower())
        else:
            allcaps = False
            break
    if allcaps:
        return ('{ALLCAPS}',) + tuple(result)
    else:
        result = []
        for c in seq:
            if c.isupper() and c not in shared.multichar_symbols:
                result.append('{CAP}')
                result.append(c.lower())
            elif c in shared.normalization_substitutions:
                result.append(shared.normalization_substitutions[c])
            else:
                result.append(c)
        return tuple(result)

def unnormalize_seq(seq):
    raise NotImplementedError()

def tokenize_word(string):
    '''Separate a string into a word and a POS-tag,
       both expressed as sequences of symbols.'''
    m = re.match(shared.compiled_patterns['word'], string)
    if m is None:
        raise Exception('Error while tokenizing word: %s' % string)
    return tuple(re.findall(shared.compiled_patterns['symbol'], m.group('word'))),\
           tuple(re.findall(shared.compiled_patterns['tag'], m.group('tag'))),\
           m.group('disamb')

class LexiconEdge:
    def __init__(self, source, target, rule, cost=0.0):
        self.source = source
        self.target = target
        self.rule = rule
        self.cost = cost
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __hash__(self):
        return self.source.__hash__() + self.target.__hash__()

class LexiconNode:
    def __init__(self, word, freq=None, vec=None):
        self.key = word
        self.word, self.tag, self.disamb = tokenize_word(word)
        self.word = normalize_seq(self.word)
# TODO deprecated
#         self.key = ''.join(self.word + self.tag) +\
#                    ((shared.format['word_disamb_sep'] + self.disamb)\
#                      if self.disamb else '')
#         self.word_tag_str = ''.join(self.word + self.tag)
        if shared.config['Features'].getfloat('word_freq_weight') > 0:
            self.freq = freq
            self.logfreq = math.log(self.freq)
        if shared.config['Features'].getfloat('word_vec_weight') > 0:
            self.vec = vec
            if self.vec is None:
                raise Exception("%s vec=None" % (self.key))
            if self.vec.shape[0] != shared.config['Features']\
                                          .getfloat('word_vec_dim'):
                raise Exception("%s dim=%d" % (self.key, self.vec.shape[0]))
        self.parent = None
        self.alphabet = None
        self.edges = []
        self.cost = 0.0
#        self.training = True
#        self.structure = None

#    def key(self):
#        return ''.join(self.word + self.tag)
    
    def __lt__(self, other):
        return self.key < other.key
    
    def __eq__(self, other):
        return isinstance(other, LexiconNode) and self.key == other.key
    
    def __str__(self):
        return self.key

    def __hash__(self):
        return self.key.__hash__()
    
    def root(self):
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def subtree(self):
        result = set([self])
        for edge in self.edges:
            result |= edge.target.subtree()
        return result
    
    def has_ancestor(self, node):
        if self.parent is None:
            return False
        else:
            if self.parent == node:
                return True
            else:
                return self.parent.has_ancestor(node)
    
    def deriving_rule(self):
        if self.prev is None:
            return None
        else:
            return self.prev.edge_label(self.key)
    
    def seq(self):
        return tuple(zip(self.word + self.tag, self.word + self.tag))
    
    def ngrams(self, n):
        # return the n-gram representation
        raise NotImplementedError()
    
    def analysis(self):
        analysis = []
        node = self
        while node.prev is not None:
            node = node.prev
            analysis.append(node.key())
        return analysis


class Lexicon:
    def __init__(self, rootdist=None, ruleset=None):
        self.nodes = {}
        self.roots = set()
        self.edges_by_rule = defaultdict(lambda: list())
        self.cost = 0.0
        self.rules_c = defaultdict(lambda: 0)
        self.transducer = None
    
    def __len__(self):
        return len(self.nodes)
    
    def __contains__(self, key):
        return self.nodes.__contains__(key)
    
    def keys(self):
        return self.nodes.keys()

    def values(self):
        return self.nodes.values()
    
    def __getitem__(self, key):
        return self.nodes[key]
    
    def __setitem__(self, key, val):
        self.nodes[key] = val
    
    def __delitem__(self, key):
        del self.nodes[key]
    
    def iter_nodes(self):
        return self.nodes.values()
    
    def iter_edges(self):
        for node in self.iter_nodes():
            for edge in node.edges:
                yield edge

    def recompute_cost(self, model):
        self.cost = sum(rt.cost for rt in self.roots) +\
            sum(edge.cost for edge in self.iter_edges()) +\
            model.null_cost()
    
    def recompute_all_costs(self, model):
        self.cost = 0
        for node in self.iter_nodes():
            node.cost = model.word_cost(node)
            self.cost += node.cost
        for edge in self.iter_edges():
            edge.cost = model.edge_cost(edge)
            self.cost += edge.cost - edge.target.cost
    
    def add_node(self, node):
        key = node.key
        self.nodes[key] = node
        self.roots.add(node)
        self.cost += node.cost
    
#     def trigram_hash(self):
#         trh = TrigramHash()
#         for node in self.nodes.values():
#             trh.add(node)
#         return trh

    # TODO full cycle detection
    def check_if_edge_possible(self, edge):
#        n1, n2 = self.nodes[edge.source], self.nodes[edge.target]
        if edge.source.parent is not None and edge.source.parent == edge.target:
            raise Exception('Cycle detected: %s, %s' % (edge.source.key, edge.target.key))
        if edge.target.parent is not None:
            raise Exception('%s has already got an ingoing edge.' % edge.target.key)
    
    def has_edge(self, edge):
        return edge in edge.source.edges

    def add_edge(self, edge):
        self.check_if_edge_possible(edge)
#        w1, w2 = self.nodes[edge.source], self.nodes[edge.target]
        edge.source.edges.append(edge)
        edge.target.parent = edge.source
        self.roots.remove(edge.target)
        self.edges_by_rule[edge.rule].append(edge)
        self.rules_c[edge.rule] += 1
        self.cost += edge.cost - edge.target.cost
    
    def remove_edge(self, edge):
        edge.target.parent = None
        self.roots.add(edge.target)
        edge.source.edges.remove(edge)
        self.edges_by_rule[edge.rule].remove(edge)
        self.rules_c[edge.rule] -= 1
        self.cost -= edge.cost - edge.target.cost
    
    def reset(self):
        self.edges_by_rule = defaultdict(lambda: list())
        for node in self.iter_nodes():
            node.parent = None
            node.edges = []
            self.roots.add(node)
    
# TODO deprecated
#     def build_transducer(self, print_progress=False):
#         self.alphabet =\
#             tuple(sorted(set(
#                 itertools.chain(*(n.word+n.tag for n in self.nodes.values()))
#             )))
#         self.transducer =\
#             algorithms.fst.binary_disjunct(
#                 [algorithms.fst.seq_to_transducer(\
#                          n.seq(), alphabet=self.alphabet)\
#                      for n in self.nodes.values()],
#                 print_progress=print_progress
#             )
    
    def save_to_file(self, filename):
        def write_subtree(fp, source, target, rule):
            line = (source.key, target.key, str(rule)) if source is not None\
                else ('', target.key, '')
            if settings.WORD_FREQ_WEIGHT > 0.0:
                line = line + (target.freq,)
            if settings.WORD_VEC_WEIGHT > 0.0:
                line = line + (settings.VECTOR_SEP.join(map(str, n2.vec)),)
            write_line(fp, line)
            for edge in target.edges:
                write_subtree(fp, target, edge.target, edge.rule)
        with open_to_write(filename) as fp:
            for rt in self.roots:
                write_subtree(fp, None, rt, None)
    
    @staticmethod
    def init_from_wordlist(filename):
        '''Create a lexicon with no edges from a wordlist.'''
        lexicon = Lexicon()
        for node_data in read_tsv_file(filename, get_wordlist_format()):
            # if the input file contains base words -> ignore them for now
            if shared.config['General'].getboolean('supervised'):
                node_data = node_data[1:]
#             try:
            lexicon.add_node(LexiconNode(*node_data))
#             except Exception:
#                 logging.getLogger('main').warning('ignoring %s' % node_data[0])
        return lexicon

    # init from file of form word_1<TAB>word_2 (extract rules)
    # TODO: change name
    @staticmethod
    def init_from_training_file(filename):
        '''Create a lexicon from a list of word pairs.'''
        lexicon = Lexicon()
        for node_1_key, node_2_key in read_tsv_file(filename, (str, str)):
            if node_1_key not in lexicon:
                lexicon.add_node(LexiconNode(node_1_key))
            if node_2_key not in lexicon:
                lexicon.add_node(LexiconNode(node_2_key))
            n1, n2 = lexicon[node_1_key], lexicon[node_2_key]
            rule = algorithms.align.extract_rule(n2, n1)
            lexicon.add_edge(LexiconEdge(n2, n1, rule))
        return lexicon

    @staticmethod
    def load_from_file(filename):
        lexicon = Lexicon()
        for line in read_tsv_file(filename, settings.LEXICON_FORMAT):
            node_data = (line[1],) + line[3:]
            lexicon.add_node(LexiconNode(*node_data))
            if line[0] and line[2]:
                lexicon.add_edge(LexiconEdge(*line[:3]))
        return lexicon
        # TODO rootdist

    def save_model(self, rules_file, lexicon_file):
        '''Save rules and lexicon.'''
        self.ruleset.save_to_file(rules_file)
        self.save_to_file(lexicon_file)

    @staticmethod
    def load_model(rules_file, lexicon_file):
        '''Load rules and lexicon.'''
        ruleset = RuleSet.load_from_file(rules_file)
        lexicon = Lexicon.load_from_file(lexicon_file, ruleset)
        return lexicon
    
