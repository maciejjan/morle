from algorithms.clustering import chinese_whispers
from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.graph import FullGraph, EdgeSet, GraphEdge
from datastruct.rules import Rule
from utils.files import full_path, read_tsv_file, open_to_write
import shared

import logging
import numpy as np
from operator import itemgetter
import random
import tqdm
from typing import List


def load_graph(filename, lexicon, threshold=0.0):
    edge_set = EdgeSet(lexicon)
    weights = []
    rules = {}
    for word_1, word_2, rule_str, edge_freq_str in read_tsv_file(filename):
        try:
            edge_freq = float(edge_freq_str)
            if edge_freq < threshold:
                continue
            if rule_str not in rules:
                rules[rule_str] = Rule.from_string(rule_str)
            edge = GraphEdge(lexicon[word_1], lexicon[word_2], rules[rule_str],
                             weight=edge_freq)
            edge_set.add(edge)
            weights.append(edge_freq)
        except ValueError:
            pass
    return FullGraph(lexicon, edge_set), np.array(weights)


def run() -> None:
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    logging.getLogger('main').info('Loading graph...')
    graph, weights = \
        load_graph('sample-edge-stats.txt',
                   lexicon,
                   threshold=shared.config['cluster'].getfloat('threshold'))
    logging.getLogger('main').info('Clustering...')
    clusters = chinese_whispers(graph, weights)
    with open_to_write('clusters.txt') as fp:
        for cluster in clusters:
            fp.write(', '.join([str(node) for node in cluster])+'\n')

