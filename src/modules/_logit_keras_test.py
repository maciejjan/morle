from datastruct.lexicon import LexiconEntry
from datastruct.graph import GraphEdge
from datastruct.rules import Rule
from models.neural import NeuralModel
from utils.files import read_tsv_file
import shared

import numpy as np
from typing import Iterable, List

# TODO
# - why is the rule :/en:ung extracted rather than :/e:u/:g?
# - ngram feature bug: multi-character symbols


def load_edges(filename):
    lexicon, rules = {}, {}
    edges, y = [], []
    for word_1, word_2, rule_str, prob in \
            read_tsv_file(filename, types=(str, str, str, float),\
                          show_progressbar=True):
        if word_1 not in lexicon:
            lexicon[word_1] = LexiconEntry(word_1)
        if word_2 not in lexicon:
            lexicon[word_2] = LexiconEntry(word_2)
        if rule_str not in rules:
            rules[rule_str] = Rule.from_string(rule_str)
        edges.append(GraphEdge(lexicon[word_1], lexicon[word_2],\
                               rules[rule_str]))
        y.append(prob)
    return edges, np.array(y)


def run():
    edges, y = load_edges(shared.filenames['sample-edge-stats'])

    model = NeuralModel(edges)
    y1 = model.y_pred.copy()
    model.fit(y)
    model.recompute_edge_prob()
    y2 = model.y_pred.copy()
    for i, edge in enumerate(edges):
        print(str(edge.source), str(edge.target), str(edge.rule),
              str(float(y[i])), str(float(y2[i])), sep='\t')

