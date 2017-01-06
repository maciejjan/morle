from datastruct.rules import Rule
from models.point import PointModel
from utils.files import read_tsv_file, file_exists
import shared

import logging

'''Fit the model using all possible edges.'''

def prepare_model():
    logging.getLogger('main').info('Loading rules...')
    rules, rule_domsizes, rule_freqs = {}, {}, {}
    filename = shared.filenames['rules-modsel']
    if not file_exists(filename):
        filename = shared.filenames['rules']
    for rule, freq, domsize in read_tsv_file(filename,\
            (str, int, int)):
        rules[rule] = Rule.from_string(rule)
        rule_domsizes[rule] = domsize
        rule_freqs[rule] = freq
    logging.getLogger('main').info('Loading edges...')
    model = PointModel(None, None)
    model.fit_ruledist(set(rules.values()))
    for rule, domsize in rule_domsizes.items():
        model.add_rule(rules[rule], domsize)
        model.rule_features[rules[rule]][0].prob = rule_freqs[rule] / domsize
    return model

def run():
    model = prepare_model()
    model.save_rules(shared.filenames['rules-fit'])

def cleanup():
    remove_file_if_exists(shared.filenames['rules-fit'])

