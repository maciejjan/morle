import algorithms.align
# import algorithms.fastss
# import algorithms.fst
import algorithms.fstfastss
# import algorithms.splrules # TODO deprecated
from datastruct.lexicon import normalize_word, Lexicon
from datastruct.rules import *
# from models.point import *
from utils.files import file_exists, read_tsv_file
# from utils.printer import *

import logging
import multiprocessing
import os.path
import subprocess
import tqdm
from typing import Any, Callable, Iterable, List, Tuple
# import threading


def load_normalized_wordlist(filename :str) -> Iterable[str]:
    results = []
    for (word,) in read_tsv_file(filename):
        results.append(normalize_word(word))
    return results


# input file: wordlist
# output file: transducer file
def compile_lexicon_transducer(
        lexicon :Lexicon = None, 
        output_file :str = None, 
        node_keys :Iterable[str] = None) -> None:

    mandatory_args = (lexicon, output_file)
    assert not any(arg is None for arg in mandatory_args)

    nodes = list(lexicon.iter_nodes())\
                if node_keys is None \
                else list(lexicon[key] for key in node_keys)
    lex_file = output_file + '.lex'
    tags = set()
    for node in nodes:
        for t in node.tag:
            tags.add(t)
    with open_to_write(lex_file) as lexfp:
        lexfp.write('Multichar_Symbols ' + 
                    ' '.join(shared.multichar_symbols + list(tags)) + '\n\n')
        lexfp.write('LEXICON Root\n')
        for node in nodes:
            lexfp.write('\t' + ''.join(node.word + node.tag) + ' # ;\n')
    # compile the lexc file
    cmd = ['hfst-lexc', full_path(lex_file), '-o', full_path(output_file)]
    subprocess.run(cmd)
    remove_file(lex_file)


def parallel_execute(
        function :Callable[..., None] = None,
        data :List[Any] = None,
        num :int = 1,
        additional_args :Tuple = ()) -> None:

    mandatory_args = (function, data, num, additional_args)
    assert not any(arg is None for arg in mandatory_args)

    # data -- partitioned equally among processes
    # function gets the following arguments: p_id, data, *args
    count = 0
    step = len(data) // num
    processes = []
    for i in range(num):
        # account for rounding error while processing the last chunk
        data_chunk = data[i*step:(i+1)*step] if i < num-1 else data[i*step:]
        p = multiprocessing.Process(target=function, 
                                    args=(i, data_chunk) + additional_args)
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()

def expand_graph(graph_file :str) -> None:
    '''Annotate graph with additional information needed for filtering:
       currently rule frequencies.'''
    min_freq = shared.config['preprocess'].getint('min_rule_freq')
    with open_to_write(graph_file + '.tmp') as graph_tmp_fp:
        for rule, wordpairs in read_tsv_file_by_key(graph_file, 3, 
                print_progress=True,
                print_msg='Expanding the graph for filtering...'):
            freq = len(wordpairs)
            if freq >= min_freq:
                for w1, w2 in wordpairs:
                    write_line(graph_tmp_fp, (w1, w2, rule, freq))
    rename_file(graph_file + '.tmp', graph_file)
    update_file_size(graph_file)

def contract_graph(graph_file :str) -> None:
    '''Remove any additional information needed for filtering.'''
    with open_to_write(graph_file + '.tmp') as graph_tmp_fp:
        for w1, w2, rule, freq in read_tsv_file(graph_file,
                print_progress=True, print_msg='Contracting the graph...'):
            write_line(graph_tmp_fp, (w1, w2, rule))
    rename_file(graph_file + '.tmp', graph_file)

def filter_max_num_rules(graph_file :str) -> None:
    logging.getLogger('main').info('filter_max_num_rules')
    sort_files(graph_file, stable=True, numeric=True, reverse=True, key=4)
    pp = progress_printer(shared.config['preprocess'].getint('max_num_rules'))
    with open_to_write(graph_file + '.tmp') as graph_fil_fp:
        num_rules = 0
        for (rule, freq), wordpairs in read_tsv_file_by_key(graph_file, (3, 4)):
            try:
                next(pp)
            except StopIteration:
                break
            if int(freq) >= shared.config['preprocess'].getint('min_rule_freq'):
                for w1, w2 in wordpairs:
                    write_line(graph_fil_fp, (w1, w2, rule, freq))
            num_rules += 1
    rename_file(graph_file + '.tmp', graph_file)
    update_file_size(graph_file)

def filter_max_edges_per_wordpair(graph_file :str) -> None:
    sort_files(graph_file, stable=True, key=(1, 2))
    with open_to_write(graph_file + '.tmp') as graph_fil_fp:
        for (word_1, word_2), edges in read_tsv_file_by_key(graph_file, (1, 2),
                print_progress=True, print_msg='filter_max_edges_per_wordpair'):
            for rule, freq in edges[:shared.config['preprocess'].getint('max_edges_per_wordpair')]:
                write_line(graph_fil_fp, (word_1, word_2, rule, freq))
    rename_file(graph_file + '.tmp', graph_file)
    sort_files(graph_file, key=3)
    sort_files(graph_file, stable=True, numeric=True, reverse=True, key=4)
    update_file_size(graph_file)

# again, because after other filters some frequencies have decreased
def filter_min_rule_freq(graph_file :str) -> None:
    with open_to_write(graph_file + '.tmp') as graph_fil_fp:
        for (rule, freq), wordpairs in read_tsv_file_by_key(graph_file, (3, 4),
                print_progress=True, print_msg='filter_min_rule_freq'):
            if len(wordpairs) >= shared.config['preprocess'].getint('min_rule_freq'):
                for word_1, word_2 in wordpairs:
                    write_line(graph_fil_fp, (word_1, word_2, rule, freq))
    rename_file(graph_file + '.tmp', graph_file)
    update_file_size(graph_file)

def run_filters(graph_file :str) -> None:
    expand_graph(graph_file)
    filter_max_num_rules(graph_file)
    filter_max_edges_per_wordpair(graph_file)
    filter_min_rule_freq(graph_file)
    contract_graph(graph_file)

def filter_rules(graph_file :str) -> None:
    '''Filter rules according to frequency.'''
    # Annotate graph with rule frequency
    # format: w1 w2 rule -> w1 w2 rule freq
    # truncate the graph file to most frequent rules
    # sort edges according to wordpair
    sort_files(graph_file + '.filtered', key=3)
    sort_files(graph_file + '.filtered', stable=True, numeric=True, reverse=True, key=4)
    remove_file(graph_file)
    remove_file(graph_file + '.tmp2')
    # cleanup files
#    rename_file(graph_file, graph_file + '.orig')
    rename_file(graph_file + '.filtered', graph_file)

def build_graph_allrules(lexicon :Lexicon, graph_file :str) -> None:
    with open_to_write(graph_file) as fp:
        for n1, n2 in algorithms.fastss.similar_words(lexicon, print_progress=True):
            for rule in algorithms.align.extract_all_rules(n1, n2):
                write_line(fp, (str(n1), str(n2), str(rule)))
                write_line(fp, (str(n2), str(n1), rule.reverse().to_string()))
    sort_files(graph_file, key=3)

def build_graph_fstfastss(lexicon, lex_tr_file, graph_file, node_keys=None):
    logging.getLogger('main').info('Building the FastSS cascade...')
    max_word_len = max([len(n.word) for n in lexicon.iter_nodes()])
    algorithms.fstfastss.build_fastss_cascade(lex_tr_file,
                                              max_word_len=max_word_len)

    def _extract_candidate_edges(p_id, words, lexicon, transducer_path,\
                                 output_file):
        sw = algorithms.fstfastss.similar_words(words, transducer_path)
        with open_to_write(output_file + '.' + str(p_id)) as outfp:
            for word_1, word_2 in sw:
                if word_1 != word_2:
                    n1, n2 = lexicon[word_1], lexicon[word_2]
                    rules = algorithms.align.extract_all_rules(n1, n2)
                    for rule in rules:
                        write_line(outfp, (n1.literal, n2.literal, str(rule)))
#                         write_line(outfp, (n2.literal, n1.literal, 
#                                            rule.reverse().to_string()))

    logging.getLogger('main').info('Building the graph...')
    if node_keys is None:
        node_keys = sorted(list(lexicon.keys()))
    transducer_path = shared.filenames['fastss-tr']
    num_processes = shared.config['preprocess'].getint('num_processes')
    parallel_execute(function=_extract_candidate_edges,
                     data=node_keys, num=num_processes,
                     additional_args=(lexicon, transducer_path, graph_file))
    outfiles = ['.'.join((graph_file, str(p_id)))\
                for p_id in range(num_processes)]
    sort_files(outfiles, outfile=graph_file, key=3, parallel=num_processes)
    for outfile in outfiles:
        remove_file(outfile)

def build_graph_from_training_edges(lexicon, training_file, graph_file):
    with open_to_write(graph_file) as fp:
        for word_1, word_2 in read_tsv_file(training_file, (str, str)):
            if word_1:
                try:
                    n1, n2 = lexicon[word_1], lexicon[word_2]
                    for rule in algorithms.align.extract_all_rules(n1, n2):
                        write_line(fp, (str(n1), str(n2), str(rule)))
                except KeyError:
                    if word_1 not in lexicon:
                        logging.getLogger('main').warning('%s not in lexicon' % word_1)

def compute_rule_domsizes(lexicon_tr_file, rules_file):
    
    def _compute_domsizes(p_id, rules, lexicon_tr, outlck, outfp):
        results = []
        # compute domsizes
        progressbar = None
        if shared.config['preprocess'].getint('num_processes') == 1:
            progressbar = tqdm.tqdm(total=len(rules))
        for rule_str, freq in rules:
            rule = Rule.from_string(rule_str)
            domsize = rule.compute_domsize(lexicon_tr)
            results.append((rule_str, freq, domsize))
            if progressbar is not None:
                progressbar.update()
        if progressbar is not None:
            progressbar.close()
        # write the results to the output file
        with outlck:
            for rule_str, freq, domsize in results:
                write_line(outfp, (rule_str, freq, domsize))
            outfp.flush()

    lexicon_tr = algorithms.fst.load_transducer(lexicon_tr_file)
    rules = [(rule_str, freq) for rule_str, freq in \
                                  read_tsv_file(rules_file, (str, int))]
    outlck = multiprocessing.Lock()
    output_file = rules_file + '.tmp'
    num_processes = shared.config['preprocess'].getint('num_processes')
    with open_to_write(output_file) as outfp:
        parallel_execute(_compute_domsizes, rules, num=num_processes,
                         additional_args=(lexicon_tr, outlck, outfp))
    rename_file(output_file, rules_file)
    sort_files(rules_file, reverse=True, numeric=True, key=2)

def run_standard():
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    logging.getLogger('main').info('Building the lexicon transducer...')
    compile_lexicon_transducer(lexicon, shared.filenames['lexicon-tr'])

    if shared.config['General'].getboolean('supervised'):
        logging.getLogger('main').info('Building graph...')
        build_graph_from_training_edges(lexicon,
                                        shared.filenames['wordlist'],
                                        shared.filenames['graph'])
    else:
        logging.getLogger('main').info('Building graph...')
#         build_graph_allrules(lexicon, shared.filenames['graph'])
        build_graph_fstfastss(lexicon, shared.filenames['lexicon-tr'], 
                              shared.filenames['graph'])

    update_file_size(shared.filenames['graph'])
    run_filters(shared.filenames['graph'])
    update_file_size(shared.filenames['graph'])
    aggregate_file(shared.filenames['graph'],\
                   shared.filenames['rules'], 3)
    update_file_size(shared.filenames['rules'])

    logging.getLogger('main').info('Computing rule domain sizes...')
    compute_rule_domsizes(shared.filenames['lexicon-tr'], 
                          shared.filenames['rules'])

def run_bipartite():
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    wordlist_left = load_normalized_wordlist(
                        shared.filenames['wordlist.left'])
    wordlist_right = load_normalized_wordlist(
                         shared.filenames['wordlist.right'])
    logging.getLogger('main').info('Building the lexicon transducers...')
    compile_lexicon_transducer(lexicon, shared.filenames['left-tr'],\
                               node_keys=wordlist_left)
    compile_lexicon_transducer(lexicon, shared.filenames['right-tr'],\
                               node_keys=wordlist_right)

    logging.getLogger('main').info('Building graph...')
    build_graph_fstfastss(lexicon, shared.filenames['right-tr'], 
                          shared.filenames['graph'],
                          node_keys=wordlist_left)

    update_file_size(shared.filenames['graph'])
    run_filters(shared.filenames['graph'])
    update_file_size(shared.filenames['graph'])
    aggregate_file(shared.filenames['graph'],\
                   shared.filenames['rules'], 3)
    update_file_size(shared.filenames['rules'])

    logging.getLogger('main').info('Computing rule domain sizes...')
    compute_rule_domsizes(shared.filenames['left-tr'], 
                          shared.filenames['rules'])

### MAIN FUNCTIONS ###

def run():
    if file_exists(shared.filenames['wordlist.left']) and\
       file_exists(shared.filenames['wordlist.right']):
        run_bipartite()
    elif file_exists(shared.filenames['wordlist']):
        run_standard()
    else:
        raise RuntimeError('No input file supplied!')

def cleanup():
    remove_file_if_exists(shared.filenames['rules'])
    remove_file_if_exists(shared.filenames['lexicon-tr'])
    remove_file_if_exists(shared.filenames['graph'])

