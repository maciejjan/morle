# evaluates all inflection tasks at once
# * lemmatization: word + tag -> lemma + lemma_tag
#   acceptor(word . tag) .o. inv(rules) .o. lexicon
# * tagging: word -> tag
#   in_proj((acceptor(word) . tag_acceptor()) .o. inv(rules) .o. lexicon)
# * generation: lemma + lemma_tag + tag -> word
#   out_proj(acceptor(lemma . tag) .o. rules .o. (word_acceptor() . acceptor(tag)))

# also for untagged data:
# * lemmatization: word -> lemma

# result of each trial: integer
# 0 - the correct result not found
# n > 0 - the correct result found as nth

# format results:
# n   lemmatization   tagging   generation
# (numbers for each n)

from collections import namedtuple
import libhfst

import algorithms.fst
from datastruct.lexicon import *
#from models.point import *
from utils.files import *
import shared
#
#import logging
#from operator import itemgetter
#import sys

def prepare_automata():
    Automata = namedtuple('Automata', 
                          ['lexicon', 'rules', 'word_acc', 
                           'tag_acc', 'word_gen', 'tag_gen'])
    lexicon_tr = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    rules_tr.invert()
    rules_tr.compose(lexicon_tr)
#    rules_tr.remove_epsilons()
#    rules_tr.determinize()
    rules_tr.minimize()
    rules_tr.convert(libhfst.HFST_OLW_TYPE)
    return Automata(lexicon_tr, rules_tr, None, None, None, None) 

def result_index(correct, results):
    try:
        return results.index(correct)+1
    except ValueError:
        return 0
    
def words_from_paths(paths):
    return [word.replace(libhfst.EPSILON, '') 
            for word, cost in paths]

def eval_lemmatize(word, base, automata):
    results = words_from_paths(automata.rules.lookup(word))
    return result_index(base, results)

def eval_tag(word, base, automata):
    return 0
    raise NotImplementedError()

def eval_generate(word, base, automata):
    return 0
    raise NotImplementedError()

def run():
    automata = prepare_automata()
    with open_to_write(shared.filenames['eval.report']) as fp:
        for base, word in read_tsv_file(shared.filenames['eval.wordlist'], 
                                        types=(str, str), print_progress=True,
                                        print_msg='Evaluating...'):
            if not base.strip():
                base = word
            write_line(fp, (word, base,
                            eval_lemmatize(word, base, automata),
                            eval_tag(word, base, automata),
                            eval_generate(word, base, automata)))

