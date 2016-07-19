from utils.files import *

from collections import defaultdict
from operator import itemgetter
import random

def load_graph(filename):
    entering = {}
    for i, (v1, v2, r, w) in enumerate(read_tsv_file(filename)):
        if i == 0:
            continue
        if v1 not in entering:
            entering[v1] = []
        if v2 not in entering:
            entering[v2] = []
        entering[v2].append((v1, float(w)))
    return entering

def save_clusters(clusters, filename):
    with open_to_write(filename) as fp:
        for cl in clusters:
            write_line(fp, (', '.join(cl),))

def chinwhisp(entering):
    cluster = {}
    for i, v in enumerate(entering):
        cluster[v] = i

    changes = True
    vertices = list(entering)
    while changes:
        changes = False
        random.shuffle(vertices)
        for v in vertices:
            if entering[v]:
                cl_candidates = defaultdict(lambda: 0)
                for v1, w in entering[v]:
                    cl_candidates[cluster[v1]] += w
                new_cluster = max(cl_candidates.items(), key=itemgetter(1))[0]
                if new_cluster != cluster[v]:
                    changes = True
                cluster[v] = new_cluster

    by_cluster = {}
    for v, cl in cluster.items():
        if cl not in by_cluster:
            by_cluster[cl] = []
        by_cluster[cl].append(v)
    return list(by_cluster.values())

def run():
    graph = load_graph('edges_sample.txt.1')
    clusters = chinwhisp(graph)
    save_clusters(clusters, 'clusters.txt')

