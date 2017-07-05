import algorithms.em
from datastruct.graph import FullGraph
from datastruct.lexicon import Lexicon
from datastruct.rules import Rule
# from models.point import PointModel
from models.neural import ModelSuite
from utils.files import file_exists, read_tsv_file
import shared

from collections import defaultdict
import logging


def run() -> None:
    # load the lexicon
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon(shared.filenames['wordlist'])

    logging.getLogger('main').info('Loading rules...')
    rules_file = shared.filenames['rules-modsel']
    if not file_exists(rules_file):
        rules_file = shared.filenames['rules']
    rules = [(Rule.from_string(rule_str), domsize) \
             for rule_str, domsize in \
                 read_tsv_file(rules_file, (str, int))]

    # load the full graph
    graph_file = shared.filenames['graph-modsel']
    if not file_exists(graph_file):
        graph_file = shared.filenames['graph']
    logging.getLogger('main').info('Loading the graph...')
    full_graph = FullGraph(lexicon)
    full_graph.load_edges_from_file(graph_file)

    # count rule frequencies in the full graph
    rule_freq = defaultdict(lambda: 0)
    for edge in full_graph.iter_edges():
        rule_freq[edge.rule] += 1

    # initialize a PointModel
    logging.getLogger('main').info('Initializing the model...')
    model = ModelSuite(full_graph, dict(rules))
#     model = PointModel()
#     model.fit_rootdist(lexicon.entries())
#     model.fit_ruledist(rule for (rule, domsize) in rules)
#     for rule, domsize in rules:
#         model.add_rule(rule, domsize, freq=rule_freq[rule])

    algorithms.em.softem(full_graph, model)

