from datastruct.lexicon import Lexicon
from utils.files import read_tsv_file
import shared

import networkx as nx
import time

def load_graph(filename, lexicon):
    graph = nx.MultiDiGraph()
    for w1, w2, rule in read_tsv_file(filename, show_progressbar=True):
        graph.add_edge(lexicon[w1], lexicon[w2], rule)
    return graph

def run():
    lexicon = Lexicon(shared.filenames['wordlist'])
    graph = load_graph(shared.filenames['graph'], lexicon)
    print(graph.number_of_edges())
    print('done')
    time.sleep(10)

