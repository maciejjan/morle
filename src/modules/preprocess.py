import algorithms.align
import algorithms.fastss
import algorithms.fst
import algorithms.fstfastss
# import algorithms.splrules # TODO deprecated
from datastruct.lexicon import *
from datastruct.rules import *
# from models.point import *
from utils.files import *
from utils.printer import *

import logging
import multiprocessing
import os.path
import subprocess
import tqdm
# import threading

# input file: wordlist
# output file: transducer file
def build_lexicon_transducer(lexicon, output_file):
    # build the lexc file
    lex_file = output_file + '.lex'
    tags = set()
    for node in lexicon.iter_nodes():
        for t in node.tag:
            tags.add(t)
    with open_to_write(lex_file) as lexfp:
        lexfp.write('Multichar_Symbols ' + 
                    ' '.join(shared.multichar_symbols + list(tags)) + '\n\n')
        lexfp.write('LEXICON Root\n')
        for node in lexicon.iter_nodes():
            lexfp.write('\t' + ''.join(node.word + node.tag) + ' # ;\n')
    # compile the lexc file
    cmd = ['hfst-lexc', full_path(lex_file), '-o', full_path(output_file)]
    subprocess.run(cmd)
    remove_file(lex_file)

def partition_data_for_parallelization(size):
    '''Divides the data into equal chunks for threads.
       Takes only the size of the data as parameter
       and returns the indices for each thread.'''
    num_processes = shared.config['preprocess'].getint('num_processes')
    step = size // num_processes
    cur_idx = 0
    for i in range(num_processes-1):
        yield (cur_idx, cur_idx+step)
        cur_idx += step
    # take account for the rounding error in the last chunk
    yield (cur_idx, size)

def expand_graph(graph_file):
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

def contract_graph(graph_file):
    '''Remove any additional information needed for filtering.'''
    with open_to_write(graph_file + '.tmp') as graph_tmp_fp:
        for w1, w2, rule, freq in read_tsv_file(graph_file,
                print_progress=True, print_msg='Contracting the graph...'):
            write_line(graph_tmp_fp, (w1, w2, rule))
    rename_file(graph_file + '.tmp', graph_file)

def filter_max_num_rules(graph_file):
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

def filter_max_edges_per_wordpair(graph_file):
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
def filter_min_rule_freq(graph_file):
    with open_to_write(graph_file + '.tmp') as graph_fil_fp:
        for (rule, freq), wordpairs in read_tsv_file_by_key(graph_file, (3, 4),
                print_progress=True, print_msg='filter_min_rule_freq'):
            if len(wordpairs) >= shared.config['preprocess'].getint('min_rule_freq'):
                for word_1, word_2 in wordpairs:
                    write_line(graph_fil_fp, (word_1, word_2, rule, freq))
    rename_file(graph_file + '.tmp', graph_file)
    update_file_size(graph_file)

def run_filters(graph_file):
    expand_graph(graph_file)
    filter_max_num_rules(graph_file)
    filter_max_edges_per_wordpair(graph_file)
    filter_min_rule_freq(graph_file)
    contract_graph(graph_file)

def filter_rules(graph_file):
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

def build_graph_allrules(lexicon, graph_file):
    with open_to_write(graph_file) as fp:
        for n1, n2 in algorithms.fastss.similar_words(lexicon, print_progress=True):
            for rule in algorithms.align.extract_all_rules(n1, n2):
                write_line(fp, (str(n1), str(n2), str(rule)))
                write_line(fp, (str(n2), str(n1), rule.reverse().to_string()))
    sort_files(graph_file, key=3)


def build_graph_fstfastss(lexicon, lexicon_tr_file, graph_file):
    logging.getLogger('main').info('Building the FastSS cascade...')
    max_word_len = max([len(n.seq()) for n in lexicon.iter_nodes()])
    algorithms.fstfastss.build_fastss_cascade(lexicon_tr_file,
                                              max_word_len=max_word_len)

    def _extract_candidate_edges(lexicon, words, transducer_path, output_file):
        sw = algorithms.fstfastss.similar_words(words, transducer_path)
        with open_to_write(output_file) as outfp:
            for word_1, word_2 in sw:
                if word_1 < word_2:
                    n1, n2 = lexicon[word_1], lexicon[word_2]
                    rules = algorithms.align.extract_all_rules(n1, n2)
                    for rule in rules:
                        write_line(outfp, (n1.literal, n2.literal, str(rule)))
                        write_line(outfp, (n2.literal, n1.literal, 
                                           rule.reverse().to_string()))

    logging.getLogger('main').info('Building the graph...')
    words = sorted(list(lexicon.keys()))
    transducer_path = shared.filenames['fastss-tr']
    processes, outfiles = [], []
#     progressbar = tqdm.tqdm(total=len(words))
    for p_id, (i, j) in\
            enumerate(partition_data_for_parallelization(len(words))):
#         print(p_id, i, j)
        outfile = graph_file + '.' + str(p_id)
        p = multiprocessing.Process(target=_extract_candidate_edges,
                                    args=(lexicon, words[i:j],
                                          transducer_path, outfile))
        processes.append(p)
        outfiles.append(outfile)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    sort_files(outfiles, outfile=graph_file, key=3, parallel=len(processes))
    for outfile in outfiles:
        remove_file(outfile)

def build_graph_bipartite(lexicon_left, lexicon_right, graph_file):
    with open_to_write(graph_file) as fp:
        for n1, n2 in algorithms.fastss.similar_words_bipartite(
                        lexicon_left, lexicon_right, print_progress=True):
            for rule in algorithms.align.extract_all_rules(n1, n2):
                write_line(fp, (str(n1), str(n2), str(rule)))

def build_graph_bipartite_fastss(lexicon_left, lexicon_right, graph_file):
    # TODO one lexicon, two different word lists
    # but: build left and right transducers, because we need them
    # goal: integrate into normal preprocess
    raise NotImplementedError() # TODO

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
    
    def _compute_domsizes(lexicon_tr, rules, outlck, outfp):
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
    with open_to_write(output_file) as outfp:
        processes = []
        for i, j in partition_data_for_parallelization(len(rules)):
            p = multiprocessing.Process(
                    target=_compute_domsizes,
                    args=(lexicon_tr, rules[i:j], outlck, outfp))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    rename_file(output_file, rules_file)
    sort_files(rules_file, reverse=True, numeric=True, key=2)

def run_standard():
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    logging.getLogger('main').info('Building the lexicon transducer...')
    build_lexicon_transducer(lexicon, shared.filenames['lexicon-tr'])

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
    # TODO
    # build the lexicon from left and right wordlist together
    # build the FastSS cascade from right wordlist
    # extract similar words for the words from the left wordlist

    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    logging.getLogger('main').info('Building the lexicon transducer...')
    # TODO build transducer: for the left lexicon (roots.fsm?)
    build_lexicon_transducer(lexicon, shared.filenames['lexicon-tr'])

    if shared.config['General'].getboolean('supervised'):
        logging.getLogger('main').info('Building graph...')
        build_graph_from_training_edges(lexicon,
                                        shared.filenames['wordlist'],
                                        shared.filenames['graph'])
    else:
        logging.getLogger('main').info('Building graph...')
#         build_graph_allrules(lexicon, shared.filenames['graph'])
        # TODO build the cascade for the right transducer
        # TODO lookup the words from the left transducer
        build_graph_fstfastss(lexicon, shared.filenames['lexicon-tr'], 
                              shared.filenames['graph'])

    # TODO the rest unchanged
    update_file_size(shared.filenames['graph'])
    run_filters(shared.filenames['graph'])
    update_file_size(shared.filenames['graph'])
    aggregate_file(shared.filenames['graph'],\
                   shared.filenames['rules'], 3)
    update_file_size(shared.filenames['rules'])

    logging.getLogger('main').info('Computing rule domain sizes...')
    compute_rule_domsizes(shared.filenames['lexicon-tr'], 
                          shared.filenames['rules'])

### MAIN FUNCTIONS ###

def run():
    if file_exists(shared.filenames['wordlist.left']) and\
       file_exists(shared.filenames['wordlist.right']):
        run_bipartite()
    elif file_exists(shared.filenames['wordlist']):
        run_standard()
    else:
        raise Exception('No input file supplied!')

def cleanup():
    remove_file_if_exists(shared.filenames['rules'])
    remove_file_if_exists(shared.filenames['lexicon-tr'])
    remove_file_if_exists(shared.filenames['graph'])

