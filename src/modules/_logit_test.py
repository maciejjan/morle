from datastruct.lexicon import LexiconEntry
from datastruct.graph import GraphEdge
from datastruct.rules import Rule
from models.logit import LogitModel

import numpy as np

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
    model = LogitModel(edges, rules)
    print(model.edge_probability())
    y = np.array([1, 1, 1, 0.3, 0.7, 0.05, 0.05])
    print(model.gradient(y))

