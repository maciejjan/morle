import algorithms.em
from datastruct.graph import FullGraph
from datastruct.lexicon import Lexicon
from datastruct.rules import Rule
from models.point import PointModel
from utils.files import read_tsv_file
import shared

from collections import defaultdict
import logging


def run() -> None:
    # load the lexicon
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon(shared.filenames['wordlist'])

    logging.getLogger('main').info('Loading rules...')
    rules = [(Rule.from_string(rule_str), domsize) \
             for rule_str, domsize in \
                 read_tsv_file(shared.filenames['rules'], (str, int))]

    # load the full graph
    logging.getLogger('main').info('Loading the graph...')
    full_graph = FullGraph(lexicon)
    full_graph.load_edges_from_file(shared.filenames['graph'])

    # count rule frequencies in the full graph
    rule_freq = defaultdict(lambda: 0)
    for edge in full_graph.iter_edges():
        rule_freq[edge.rule] += 1

    # initialize a PointModel
    logging.getLogger('main').info('Initializing the model...')
    model = PointModel()
    model.fit_rootdist(lexicon.entries())
    model.fit_ruledist(rule for (rule, domsize) in rules)
    for rule, domsize in rules:
        model.add_rule(rule, domsize, freq=rule_freq[rule])

    algorithms.em.softem(full_graph, model)

