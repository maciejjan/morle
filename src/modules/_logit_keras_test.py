from datastruct.lexicon import LexiconEntry
from datastruct.graph import GraphEdge
from datastruct.rules import Rule
from models.neural import NeuralModel

import numpy as np
from typing import Iterable, List

# TODO
# - move model into separate class: NeuralModel
# - test on real data and verify the convergence

def run():
    words_raw = ['machen', 'gemacht', 'macht', 'machte', 'gemachte']
    rules_raw = [':ge/en:t', ':/en:t', ':/:e', ':/en:te', 'ge:/:']
    words = [LexiconEntry(w) for w in words_raw]
    rules = [Rule.from_string(r) for r in rules_raw]
    edges = [
        GraphEdge(words[0], words[1], rules[0]),
        GraphEdge(words[0], words[2], rules[1]),
        GraphEdge(words[1], words[4], rules[2]),
        GraphEdge(words[2], words[3], rules[2]),
        GraphEdge(words[0], words[3], rules[3]),
        GraphEdge(words[1], words[2], rules[4]),
        GraphEdge(words[4], words[3], rules[4]),
    ]
    y = np.array([1, 1, 1, 0.3, 0.7, 0.05, 0.05])
    print(y)

    model = NeuralModel(edges)
    for edge in edges:
        print(str(edge.source), str(edge.target), str(edge.rule),
              model.edge_prob(edge), sep='\t')
    model.fit(y)
    print()
    model.recompute_edge_prob()
    for edge in edges:
        print(str(edge.source), str(edge.target), str(edge.rule),
              model.edge_prob(edge), sep='\t')

