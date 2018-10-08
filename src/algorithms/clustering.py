from datastruct.lexicon import LexiconEntry
from datastruct.graph import FullGraph

import logging
import numpy as np
from operator import itemgetter
import random
import tqdm
from typing import List


def chinese_whispers(graph :FullGraph, weights :np.ndarray,
                     threshold :float = 0,
                     root_weights :bool = False,
                     max_iterations :int = None) -> List[List[LexiconEntry]]:

    def _weight(edge):
        return weights[graph.edge_set.get_id(edge)]
    
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
        if max_iterations is not None and max_iterations > 0 and \
                iter_num > max_iterations:
            break
        changed = 0
        logging.getLogger('main').info('Iteration {}'.format(iter_num))
        random.shuffle(nodes_list)
        for node in tqdm.tqdm(nodes_list):
            n_id = graph.lexicon.get_id(node)
            cluster_scores = { node_cluster[n_id] : 0.0 }
            for edge in graph.ingoing_edges(node):
                if _weight(edge) > threshold:
                    src_id = graph.lexicon.get_id(edge.source)
                    if node_cluster[src_id] not in cluster_scores:
                        cluster_scores[node_cluster[src_id]] = 0
                    cluster_scores[node_cluster[src_id]] += _weight(edge)
            if root_weights:
                root_weight = 1 - sum(cluster_scores.values())
                cluster_scores[node_cluster[n_id]] += root_weight
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

