from morle.datastruct.graph import FullGraph, EdgeSet
from morle.datastruct.lexicon import Lexicon
from morle.datastruct.rules import RuleSet
from morle.models.suite import ModelSuite
from morle.utils.files import file_exists
import morle.shared as shared

import logging


def run() -> None:
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.load(shared.filenames['wordlist'])

    logging.getLogger('main').info('Loading rules...')
    rules_file = shared.filenames['rules-modsel']
    if not file_exists(rules_file):
        rules_file = shared.filenames['rules']
    rule_set = RuleSet.load(rules_file)

    edges_file = shared.filenames['graph-modsel']
    if not file_exists(edges_file):
        edges_file = shared.filenames['graph']
    logging.getLogger('main').info('Loading the graph...')
    edge_set = EdgeSet.load(edges_file, lexicon, rule_set)
    full_graph = FullGraph(lexicon, edge_set)

    # initialize a ModelSuite and save it
    logging.getLogger('main').info('Initializing the model...')
    model = ModelSuite(rule_set, lexicon = lexicon)
    model.initialize(full_graph)
    logging.getLogger('main').info('Saving the model...')
    model.save()

