import algorithms.fst
from datastruct.graph import EdgeSet
from datastruct.lexicon import Lexicon
from datastruct.rules import RuleSet
from utils.files import file_exists
import shared

import hfst
import logging
from typing import List


def extract_tag_symbols_from_rules(rule_set :RuleSet) -> List[str]:
    tags = set()
    for rule in rule_set:
        tags |= set(rule.tag_subst[0]) | set(rule.tag_subst[1])
    return sorted(list(tags))


def compute_possible_edges(lexicon :Lexicon, rule_set :RuleSet) -> EdgeSet:
    # build the transducer
    lexicon_tr = lexicon.to_fst()
    tag_symbols = extract_tag_symbols_from_rules(rule_set)
    lexicon_tr.concatenate(algorithms.fst.generator(tag_symbols))
    # TODO concatenate tag generator to lexicon_tr
    rules_tr = rule_set.to_fst()
    # TODO use algorithms.fstfastss.similar_words()
    tr = hfst.HfstTransducer(lexicon_tr)
    tr.compose(rules_tr)
    tr.determinize()
    tr.minimize()
    lexicon_tr.invert()
    tr.compose(lexicon_tr)
    tr.determinize()
    tr.minimize()
    algorithms.fst.save_transducer(tr, 'tr.fsm')
    
#     paths = tr.extract_paths(max_cycles=1)
#     for word in paths:
#         print(word, paths[word])


def run() -> None:
    # load the lexicon
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.load(shared.filenames['wordlist'])

    # load the rules
    logging.getLogger('main').info('Loading rules...')
    rules_file = shared.filenames['rules-modsel']
    if not file_exists(rules_file):
        rules_file = shared.filenames['rules']
    rule_set = RuleSet.load(rules_file)

    print(extract_tag_symbols_from_rules(rule_set))
    # TODO compute the graph of possible edges
    # TODO save the graph
    edges = compute_possible_edges(lexicon, rule_set)
#     edges.save('possible-edges.txt')

