# naive analysis:
# - lookup the word in the ruleset transducer
# - lemma cost = 0 if lemma in lexicon else rootdist(lemma)
# - also: use training data as stored lemmas

# better analysis:
# - extract_shortest_paths(query .o. inv(rules) .o. (lexicon + word_generator))
# inflected form generation:
# - extract_shortest_paths(lemma .o. rules .o. tag_acceptor)

# TODO optimize - is extracting all paths really necessary?
# lemmas: all vs. only training vs. none
# reproduce training data -> edges in lexicon!
# -> include known correspondences explicitly in the automaton (with cost 0)
#    or in the results

import algorithms.fst
from datastruct.lexicon import *
#from datastruct.rules import *
from models.point import *
import shared

import libhfst
import logging
from operator import itemgetter
import sys
#
#def load_rules():
#    return [(Rule.from_string(rule), domsize, prod)\
#            for rule, domsize, prod in\
#                read_tsv_file(shared.filenames['rules-fit'], (str, int, float))]
#
#def build_rule_transducers(rules):
##    rules.sort(reverse=True, key=itemgetter(2))
#    transducers = []
#    for rule, domsize, prod in rules:
#        rule.build_transducer(weight=-math.log(prod))
#        transducers.append(rule.transducer)
#    return transducers

def analyze(word, lexicon, rules_tr, model):
    def root_cost(node):
        return model.rootdist.cost_of_change(\
                model.extractor.extract_feature_values_from_nodes((node,)), ())

    tr = algorithms.fst.seq_to_transducer(word.seq())
    tr.compose(rules_tr)
    results = [('___',
                0 if word.key in lexicon else root_cost(word))]
    for base, cost in sum(tr.extract_paths(output='dict').values(), []):
        base = base.replace(libhfst.EPSILON, '')
        try:
            results.append((lexicon[base].key, cost))
        except KeyError:
            n_base = LexiconNode(base)
            results.append(('?'+n_base.key, cost + root_cost(n_base)))
    results.sort(key=itemgetter(1))
    return results[:shared.config['analyze'].getint('max_results')]

def run():
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    model = PointModel(lexicon)

    logging.getLogger('main').info('Ready.')
    for line in sys.stdin:
        word = LexiconNode(line.rstrip())
#        print('analyzing...')
        analysis = analyze(word, lexicon, rules_tr, model)
        for base, cost in analysis:
#            print('\t'.join(word.key(), base.key(), cost))
#            base = base.replace(libhfst.EPSILON, '')
            print('\t'.join((word.key, base, str(cost))))
#        print('finished')

def eval():
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    model = PointModel(lexicon)

    num_correct, total = 0, 0
    with open_to_write(shared.filenames['eval.report']) as outfp:
#        for word, base, freq in read_tsv_file(
        for base, word_str, freq in read_tsv_file(
                shared.filenames['eval.wordlist'], (str, str, int),
                print_progress=True, print_msg='Analyzing...'):
            if not base:
                base = '___'
            try:
                word = LexiconNode(word_str)
                analysis = analyze(word, lexicon, rules_tr, model)
                suggested_base = str(analysis[0][0])
                is_correct = (base == suggested_base)
                sym_correct = '+' if is_correct else '-'
                num_correct += 1 if is_correct else 0
                total += 1
                write_line(outfp, (word, base, suggested_base, sym_correct))
            except Exception as ex:
                print(ex)
    logging.getLogger('main').info('Accuracy: %d %%' % (num_correct * 100 / total))

