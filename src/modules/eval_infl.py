'''Inflection evaluation module.'''

from collections import defaultdict, namedtuple
import hfst
import logging

import algorithms.fst
from datastruct.lexicon import LexiconNode
from utils.files import file_exists, open_to_write, read_tsv_file, write_line
import shared

def compose_for_lemmatization(transducer):
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    transducer.compose(rules_tr)
    transducer.minimize()
    transducer.invert()
    return transducer

def compile_lemmatizer():
    logging.getLogger('main').info('Compiling the lemmatizer...')

    lemmatizer = compose_for_lemmatization(
        algorithms.fst.load_transducer(shared.filenames['rootgen-tr']))

    if shared.config['eval_infl'].getboolean('use_known_roots'):
        known_roots_lemmatizer = compose_for_lemmatization(
                                   algorithms.fst.load_transducer(
                                     shared.filenames['roots-tr']))
        lemmatizer.disjunct(known_roots_lemmatizer)

    lemmatizer.convert(hfst.ImplementationType.HFST_OLW_TYPE)
    algorithms.fst.save_transducer(lemmatizer, 
                                   shared.filenames['lemmatizer-tr'],
                                   type=hfst.ImplementationType.HFST_OLW_TYPE)
    logging.getLogger('main').info('Done.')
    return lemmatizer

def compose_for_tagging(transducer):
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    transducer.compose(rules_tr)
    transducer.minimize()
    transducer.output_project()
    for symbol in transducer.get_alphabet():
        if shared.compiled_patterns['tag'].match(symbol):
            transducer.substitute(symbol, hfst.EPSILON, 
                                  input=True, output=False)
    return transducer

def compile_tagger():
    logging.getLogger('main').info('Compiling the tagger...')
    tagger = compose_for_tagging(
        algorithms.fst.load_transducer(shared.filenames['rootgen-tr']))

    if shared.config['eval_infl'].getboolean('use_known_roots'):
        known_roots_tagger = compose_for_tagging(
                               algorithms.fst.load_transducer(
                                 shared.filenames['roots-tr']))
        tagger.disjunct(known_roots_tagger)

    tagger.convert(hfst.ImplementationType.HFST_OLW_TYPE)
    algorithms.fst.save_transducer(tagger,
                                   shared.filenames['tagger-tr'],
                                   type=hfst.ImplementationType.HFST_OLW_TYPE)
    logging.getLogger('main').info('Done.')
    return tagger

def compile_inflector():
    inflector = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    return inflector

def prepare_automata():
    Automata = namedtuple('Automata', ['lemmatizer', 'tagger', 'inflector'])

    lemmatizer, tagger, inflector = None, None, None
    if shared.config['eval_infl'].getboolean('lemmatize'):
        if shared.config['eval_infl'].getboolean('recompile') or\
                not file_exists(shared.filenames['lemmatizer-tr']):
            lemmatizer = compile_lemmatizer()
        else:
            lemmatizer = algorithms.fst.load_transducer(
                            shared.filenames['lemmatizer-tr'])

    if shared.config['eval_infl'].getboolean('tag'):
        if shared.config['eval_infl'].getboolean('recompile') or\
                not file_exists(shared.filenames['tagger-tr']):
            tagger = compile_tagger()
        else:
            tagger = algorithms.fst.load_transducer(
                         shared.filenames['tagger-tr'])

    if shared.config['eval_infl'].getboolean('inflect'):
        inflector = compile_inflector()

    return Automata(lemmatizer, tagger, inflector)

def result_index(correct, results):
    try:
        return results.index(correct)+1
    except ValueError:
        return 0
    
def words_from_paths(paths):
    return [word.replace(hfst.EPSILON, '') 
            for word, cost in paths]

def eval_lemmatize(word, base, automata):
    if not shared.config['eval_infl'].getboolean('lemmatize'):
        return 'NA'
    results = words_from_paths(automata.lemmatizer.lookup(word))
    return result_index(base, results)

def eval_tag(word, automata):
    if not shared.config['eval_infl'].getboolean('tag'):
        return 'NA'
    word_without_tag = ''.join(LexiconNode(word).word)
    results = words_from_paths(automata.tagger.lookup(word_without_tag))
    return result_index(word, results)

def eval_inflect(word, base, automata):
    if not shared.config['eval_infl'].getboolean('inflect'):
        return 'NA'
    base_tr = algorithms.fst.seq_to_transducer(LexiconNode(base).seq())
    tag_tr = algorithms.fst.tag_acceptor(LexiconNode(word).tag,
                                         automata.lemmatizer.get_alphabet())
    base_tr.compose(automata.inflector)
    base_tr.compose(tag_tr)
    base_tr.minimize()
#    base_tr.convert(hfst.HFST_OLW_TYPE)
    results = words_from_paths(base_tr.lookup(base))
    return result_index(word, results)

def print_results(results, header):
    print('\n\n%s\n' % header)
    sum_res = sum(results.values())
    for key in sorted(results.keys()):
        print('%s\t%d\t%0.1f %%' % (str(key), results[key],
                                    results[key]/sum_res*100))

def run():
    automata = prepare_automata()

    results_lem  = defaultdict(lambda: 0)
    results_tag  = defaultdict(lambda: 0)
    results_infl = defaultdict(lambda: 0)

    with open_to_write(shared.filenames['eval.report']) as fp:
        for base, word in read_tsv_file(shared.filenames['eval.wordlist'], 
                                        types=(str, str), print_progress=True,
                                        print_msg='Evaluating...'):
            if not base.strip():
                base = word
            r_lem = eval_lemmatize(word, base, automata)
            r_tag = eval_tag(word, automata)
            r_infl = eval_inflect(word, base, automata)

            results_lem[r_lem] += 1
            results_tag[r_tag] += 1
            results_infl[r_infl] += 1

            write_line(fp, (word, base, r_lem, r_tag, r_infl))

    print_results(results_lem, 'LEMMATIZATION RESULTS')
    print_results(results_tag, 'TAGGING RESULTS')
    print_results(results_infl, 'INFLECTION RESULTS')

