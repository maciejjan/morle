from datastruct.lexicon import LexiconEntry
from datastruct.graph import GraphEdge
from datastruct.rules import Rule
from models.neural import NeuralModel
from models.logit import LogitModel
from models.baseline import BaselineModel
from utils.files import read_tsv_file
import shared

import numpy as np
from operator import itemgetter
from typing import Iterable, List

# TODO
# - why is the rule :/en:ung extracted rather than :/e:u/:g?
# - max_edges_per_wordpair: take the most general rule
#   and the most general rule without infix
# TODO calculate the total benefit in entropy
# TODO the root node and the rule deriving root words


def load_edges(filename):
    lexicon, rules = {}, {}
    result = []
    for word_1, word_2, rule_str, prob in \
            read_tsv_file(filename, types=(str, str, str, float),\
                          show_progressbar=True):
        if word_1 not in lexicon:
            lexicon[word_1] = LexiconEntry(word_1)
        if word_2 not in lexicon:
            lexicon[word_2] = LexiconEntry(word_2)
        if rule_str not in rules:
            rules[rule_str] = Rule.from_string(rule_str)
        result.append((GraphEdge(lexicon[word_1], lexicon[word_2],\
                                 rules[rule_str]), prob))
    return result

def edge_prob_vector(model, edges):
    result = np.empty((len(edges),))
    for i, edge in enumerate(edges):
        result[i] = model.edge_prob(edge)
    return result

def experiment(model_class, edges_freq):
    edges = list(map(itemgetter(0), edges_freq))
    model = None
    if model_class == 'neural':
        model = NeuralModel(edges)
    elif model_class == 'logit':
        model = LogitModel(edges)
    elif model_class == 'baseline':
        model = BaselineModel(edges)
    model.fit_to_sample(edges_freq)
    model.recompute_edge_prob()
    return edge_prob_vector(model, edges)

def loss(x, y):
    return -np.nansum(y*np.log(x) + (1-y)*np.log(1-x))

def run():
    edges_freq = load_edges(shared.filenames['sample-edge-stats'])
    y = np.array(list(map(itemgetter(1), edges_freq)))

    neural_prob = experiment('neural', edges_freq)
    logit_prob = experiment('logit', edges_freq)
    baseline_prob = experiment('baseline', edges_freq)
    print('Loss function:')
    print('neural:', loss(neural_prob, y))
    print('logit:', loss(logit_prob, y))
    print('baseline:', loss(baseline_prob, y))
    print('perfect:', loss(y, y))

