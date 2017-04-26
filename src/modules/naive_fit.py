from datastruct.rules import Rule
from models.point import PointModel
from utils.files import read_tsv_file, file_exists
import shared

from collections import defaultdict
import logging


'''Fit the model using all possible edges (ignoring the tree constraints).'''


def run() -> None:
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
    logging.getLogger('main').info('Computing rule frequencies...')
    rule_freq = defaultdict(lambda: 0)
    for word_1, word_2, rule_str in read_tsv_file(graph_file):
        rule_freq[rule_str] += 1

    # initialize a PointModel
    model = PointModel()
    for rule, domsize in rules:
        model.add_rule(rule, domsize, freq=rule_freq[str(rule)])

    logging.getLogger('main').info('Saving rules...')
    model.save_rules_to_file(shared.filenames['rules-fit'])

