from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.graph import FullGraph, EdgeSet, GraphEdge
from datastruct.rules import Rule
from utils.files import full_path, read_tsv_file
import shared

import logging
import numpy as np
from operator import itemgetter
import random
import tqdm
from typing import List


def load_graph(filename, lexicon):
    edge_set = EdgeSet(lexicon)
    rules = {}
    for word_1, word_2, rule_str, edge_freq_str in read_tsv_file(filename):
        try:
            edge_freq = float(edge_freq_str)
            if rule_str not in rules:
                rules[rule_str] = Rule.from_string(rule_str)
            edge = GraphEdge(lexicon[word_1], lexicon[word_2], rules[rule_str],
                             weight=edge_freq)
            edge_set.add(edge)
        except ValueError:
            pass
    return FullGraph(lexicon, edge_set)


def chinese_whispers(graph :FullGraph) -> List[List[LexiconEntry]]:
    # initialize
    node_cluster = np.empty(graph.number_of_nodes())
    for node in graph:
        i = graph.lexicon.get_id(node)
        node_cluster[i] = i
    # run the clustering
    changed = 0
    iter_num = 0
    nodes_list = list(graph.nodes_iter())
    while True:
        iter_num += 1
        changed = 0
        logging.getLogger('main').info('Iteration {}'.format(iter_num))
        random.shuffle(nodes_list)
        for node in tqdm.tqdm(nodes_list):
            n_id = graph.lexicon.get_id(node)
            cluster_scores = { node_cluster[n_id] : 0.0 }
            for edge in graph.ingoing_edges(node):
                if edge.attr['weight'] > 0:
                    src_id = graph.lexicon.get_id(edge.source)
                    if node_cluster[src_id] not in cluster_scores:
                        cluster_scores[node_cluster[src_id]] = 0
                    cluster_scores[node_cluster[src_id]] += edge.attr['weight']
            cluster_id = max(cluster_scores.items(), key=itemgetter(1))[0]
            if cluster_id != node_cluster[n_id]:
                node_cluster[n_id] = cluster_id
                changed += 1
        logging.getLogger('main').info('changed nodes: {}'.format(changed))
        if changed == 0:
            break
    # retrieve the clusters
    clusters_hash = {}
    for i, cl_id in enumerate(node_cluster):
        if cl_id not in clusters_hash:
            clusters_hash[cl_id] = []
        clusters_hash[cl_id].append(graph.lexicon[i])
    return sorted(list(clusters_hash.values()))


def run() -> None:
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    logging.getLogger('main').info('Loading graph...')
    graph = load_graph(full_path('sample-edge-stats.txt'), lexicon)
    logging.getLogger('main').info('Clustering...')
    clusters = chinese_whispers(graph)
    for cluster in clusters:
        print(', '.join([str(node) for node in cluster]))

