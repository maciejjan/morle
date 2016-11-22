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
from utils.files import *
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
    lemmatizer.convert(libhfst.HFST_OLW_TYPE)
    algorithms.fst.save_transducer(lemmatizer, 
                                   shared.filenames['lemmatizer-tr'],
                                   type=libhfst.HFST_OLW_TYPE)
    logging.getLogger('main').info('Done.')
    return lemmatizer

def compose_for_tagging(transducer):
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    transducer.compose(rules_tr)
    transducer.minimize()
    transducer.output_project()
    tag_absorber = algorithms.fst.tag_absorber(rules_tr.get_alphabet())
    transducer.compose(tag_absorber)
    transducer.minimize()
    transducer.invert()
    return transducer

def compile_tagger():
    logging.getLogger('main').info('Compiling the tagger...')
    tagger = compose_for_tagging(
        algorithms.fst.load_transducer(shared.filenames['rootgen-tr']))
    tagger.convert(libhfst.HFST_OLW_TYPE)
    algorithms.fst.save_transducer(tagger,
                                   shared.filenames['tagger-tr'],
                                   type=libhfst.HFST_OLW_TYPE)
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
    return [word.replace(libhfst.EPSILON, '') 
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
#    base_tr.convert(libhfst.HFST_OLW_TYPE)
    results = words_from_paths(base_tr.lookup(base))
    return result_index(word, results)

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
                            eval_tag(word, automata),
                            eval_inflect(word, base, automata)))

