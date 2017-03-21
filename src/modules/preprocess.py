import algorithms.align
import algorithms.fstfastss
from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.rules import Rule
from utils.files import aggregate_file, file_exists, full_path, \
                        open_to_write, read_tsv_file, read_tsv_file_by_key, \
                        remove_file, remove_file_if_exists, rename_file, \
                        sort_file, update_file_size, write_line, \
                        write_tsv_file
import shared

from hfst import HfstTransducer, compile_lexc_file
import logging
import multiprocessing
import os.path
# import queue
import subprocess
import time
import tqdm
from typing import Any, Callable, Iterable, List, Tuple
from typing.io import TextIO


def load_normalized_wordlist(filename :str) -> Iterable[str]:
    results = []
    for (word,) in read_tsv_file(filename):
        results.append(LexiconEntry(word).normalized)
    return results


# input file: wordlist
# output file: transducer file
# TODO return the transducer instead of saving it
def compile_lexicon_transducer(entries :List[LexiconEntry]) -> HfstTransducer:
    lexc_file = shared.filenames['lexicon-tr'] + '.lex'
    tags = set()
    for entry in entries:
        for t in entry.tag:
            tags.add(t)
    with open_to_write(lexc_file) as lexfp:
        lexfp.write('Multichar_Symbols ' + 
                    ' '.join(shared.multichar_symbols + list(tags)) + '\n\n')
        lexfp.write('LEXICON Root\n')
        for entry in entries:
            lexfp.write('\t' + entry.symstr + ' # ;\n')
    # compile the lexc file
#     cmd = ['hfst-lexc', full_path(lex_file), '-o', full_path(output_file)]
#     subprocess.run(cmd)
    transducer = compile_lexc_file(lexc_file)
    remove_file(lexc_file)
    return transducer


def parallel_execute(function :Callable[..., None] = None,
                     data :List[Any] = None,
                     num :int = 1,
                     additional_args :Tuple = (),
                     show_progressbar :bool = False) -> Iterable:

    mandatory_args = (function, data, num, additional_args)
    assert not any(arg is None for arg in mandatory_args)

    # partition the data into chunks for each process
    step = len(data) // num
    data_chunks = []
    i = 0
    while i < num-1:
        data_chunks.append(data[i*step:(i+1)*step])
        i += 1
    # account for rounding error while processing the last chunk
    data_chunks.append(data[i*step:])

    queue = multiprocessing.Queue(10000)
    queue_lock = multiprocessing.Lock()

    def _output_fun(x):
        successful = False
        queue_lock.acquire()
        try:
            queue.put_nowait(x)
            successful = True
        except Exception:
            successful = False
        finally:
            queue_lock.release()
        if not successful:
            time.sleep(1)       # wait for the queue to be emptied
            _output_fun(x)

    processes, joined = [], []
    for i in range(num):
        p = multiprocessing.Process(target=function,
                                    args=(data_chunks[i], _output_fun) +\
                                         additional_args)
        p.start()
        processes.append(p)
        joined.append(False)

    progressbar, state = None, None
    if show_progressbar:
        progressbar = tqdm.tqdm(total=len(data))
    while not all(joined):
        count = 0
        queue_lock.acquire()
        try:
            while not queue.empty():
                yield queue.get()
                count += 1
        finally:
            queue_lock.release()
        if show_progressbar:
            progressbar.update(count)
        for i, p in enumerate(processes):
            if not p.is_alive():
                p.join()
                joined[i] = True
    if show_progressbar:
        progressbar.close()


def expand_graph(graph_file :str) -> None:
    '''Annotate graph with additional information needed for filtering:
       currently rule frequencies.'''
    min_freq = shared.config['preprocess'].getint('min_rule_freq')
    with open_to_write(graph_file + '.tmp') as graph_tmp_fp:
        logging.getLogger('main').info('Expanding the graph for filtering...')
        for rule, wordpairs in read_tsv_file_by_key(graph_file, 3, 
                                                    show_progressbar=True):
            freq = len(wordpairs)
            if freq >= min_freq:
                for w1, w2 in wordpairs:
                    write_line(graph_tmp_fp, (w1, w2, rule, freq))
    rename_file(graph_file + '.tmp', graph_file)
    update_file_size(graph_file)


def contract_graph(graph_file :str) -> None:
    '''Remove any additional information needed for filtering.'''
    with open_to_write(graph_file + '.tmp') as graph_tmp_fp:
        logging.getLogger('main').info('Contracting the graph...')
        for w1, w2, rule, freq in read_tsv_file(graph_file,
                                                show_progressbar=True):
            write_line(graph_tmp_fp, (w1, w2, rule))
    rename_file(graph_file + '.tmp', graph_file)


def filter_max_num_rules(graph_file :str) -> None:
    logging.getLogger('main').info('filter_max_num_rules')
    sort_file(graph_file, stable=True, numeric=True, reverse=True, key=4)
    max_num_rules = shared.config['preprocess'].getint('max_num_rules')
    min_rule_freq = shared.config['preprocess'].getint('min_rule_freq')
    progressbar = tqdm.tqdm(total=max_num_rules)
    with open_to_write(graph_file + '.tmp') as graph_fil_fp:
        num_rules = 0
        for key, wordpairs in read_tsv_file_by_key(graph_file, (3, 4)):
            rule, freq = key
            num_rules += 1
            progressbar.update()
            if int(freq) >= min_rule_freq:
                for wordpair in wordpairs:
                    w1, w2 = wordpair
                    write_line(graph_fil_fp, (w1, w2, rule, freq))
            if num_rules >= max_num_rules:
                break
    progressbar.close()
    rename_file(graph_file + '.tmp', graph_file)
    update_file_size(graph_file)


def filter_max_edges_per_wordpair(graph_file :str) -> None:
    logging.getLogger('main').info('filter_max_edges_per_wordpair')
    sort_file(graph_file, stable=True, key=(1, 2))
    max_edges_per_wordpair = \
        shared.config['preprocess'].getint('max_edges_per_wordpair')
    with open_to_write(graph_file + '.tmp') as graph_fil_fp:
        for (word_1, word_2), edges in read_tsv_file_by_key(graph_file, (1, 2),
                show_progressbar=True):
            for rule, freq in edges[:max_edges_per_wordpair]:
                write_line(graph_fil_fp, (word_1, word_2, rule, freq))
    rename_file(graph_file + '.tmp', graph_file)
    sort_file(graph_file, key=3)
    sort_file(graph_file, stable=True, numeric=True, reverse=True, key=4)
    update_file_size(graph_file)


# again, because after other filters some frequencies have decreased
def filter_min_rule_freq(graph_file :str) -> None:
    logging.getLogger('main').info('filter_min_rule_freq')
    min_rule_freq = shared.config['preprocess'].getint('min_rule_freq')
    with open_to_write(graph_file + '.tmp') as graph_fil_fp:
        for (rule, freq), wordpairs in read_tsv_file_by_key(graph_file, (3, 4),
                show_progressbar=True):
            if len(wordpairs) >= min_rule_freq:
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
    sort_file(graph_file + '.filtered', key=3)
    sort_file(graph_file + '.filtered', stable=True,
              numeric=True, reverse=True, key=4)
    remove_file(graph_file)
    # cleanup files
#    rename_file(graph_file, graph_file + '.orig')
    rename_file(graph_file + '.filtered', graph_file)


def build_graph_fstfastss(
        lexicon :Lexicon,
        lex_tr_file :str,
        words :List[str] = None) -> Iterable[Tuple[str, str, str]]:

    logging.getLogger('main').info('Building the FastSS cascade...')
    max_word_len = max([len(e.word) for e in lexicon.entries()])
    algorithms.fstfastss.build_fastss_cascade(lex_tr_file,
                                              max_word_len=max_word_len)

    def _extract_candidate_edges(words :Iterable[str],
                                 output_fun :Callable[..., None],
                                 lexicon :Lexicon,
                                 transducer_path :str) -> None:
        sw = algorithms.fstfastss.similar_words(words, transducer_path)
#         with open_to_write(output_file + '.' + str(p_id)) as outfp:
        for word_1, simwords in sw:
            v1_list = lexicon.get_by_symstr(word_1)
            for v1 in v1_list:
                results_for_v1 = []
                for word_2 in simwords:
                    for v2 in lexicon.get_by_symstr(word_2):
                        if v1 != v2:
                            rules = algorithms.align.extract_all_rules(v1, v2)
                            for rule in rules:
                                results_for_v1.append((v2.literal, str(rule)))
                output_fun((v1.literal, results_for_v1))

#     logging.getLogger('main').info('Building the graph...')
    if words is None:
        words = sorted(list(set(e.symstr for e in lexicon.entries())))
    transducer_path = shared.filenames['fastss-tr']
    num_processes = shared.config['preprocess'].getint('num_processes')
    extractor = parallel_execute(function=_extract_candidate_edges,
                                 data=words, num=num_processes,
                                 additional_args=(lexicon, transducer_path),
                                 show_progressbar=True)
    for word_1, edges in extractor:
        for word_2, rule_str in edges:
            yield (word_1, word_2, rule_str)
    # TODO sort the resulting file

#     outfiles = ['.'.join((graph_file, str(p_id)))\
#                 for p_id in range(num_processes)]
#     sort_files(outfiles, outfile=graph_file, key=3, parallel=num_processes)
#     for outfile in outfiles:
#         remove_file(outfile)


# TODO refactor
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


def compute_rule_domsizes(lexicon_tr :HfstTransducer,
                          rules :Iterable[Rule]) -> Iterable[Tuple[Rule, int]]:
    
    def _compute_domsizes(rules :Iterable[Rule], 
                          output_fun :Callable[..., None],
                          lexicon_tr :HfstTransducer) -> None:
        for rule in rules:
            output_fun((rule, rule.compute_domsize(lexicon_tr)))

    num_processes = shared.config['preprocess'].getint('num_processes')
    results = parallel_execute(_compute_domsizes, rules, num=num_processes,
                               additional_args=(lexicon_tr,),
                               show_progressbar=True)
    for rule, domsize in results:
        yield rule, domsize
#     rename_file(output_file, rules_file)
#     sort_file(rules_file, reverse=True, numeric=True, key=2)


def run_standard() -> None:
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon(filename=shared.filenames['wordlist'])
    logging.getLogger('main').info('Building the lexicon transducer...')
    lexicon_tr = compile_lexicon_transducer(lexicon.entries())
    algorithms.fst.save_transducer(lexicon_tr, shared.filenames['lexicon-tr'])

    if shared.config['General'].getboolean('supervised'):
        logging.getLogger('main').info('Building graph...')
        build_graph_from_training_edges(lexicon,
                                        shared.filenames['wordlist'],
                                        shared.filenames['graph'])
    else:
        logging.getLogger('main').info('Building graph...')
#         build_graph_allrules(lexicon, shared.filenames['graph'])
        write_tsv_file(shared.filenames['graph'],
                       build_graph_fstfastss(lexicon, 
                                             shared.filenames['lexicon-tr']))
        sort_file(shared.filenames['graph'], key=3)

    update_file_size(shared.filenames['graph'])
    run_filters(shared.filenames['graph'])
    update_file_size(shared.filenames['graph'])
    aggregate_file(shared.filenames['graph'],\
                   shared.filenames['rules'], 3)
    update_file_size(shared.filenames['rules'])

    # write rules file
    logging.getLogger('main').info('Computing rule frequencies...')
    rules = []
    rule_freq = {}
    for rule_str, edges in read_tsv_file_by_key(shared.filenames['graph'], 
                                                key=3, show_progressbar=True):
        rule_freq[rule_str] = len(edges)
        rules.append(Rule.from_string(rule_str))
    logging.getLogger('main').info('Computing rule domain sizes...')
    write_tsv_file(shared.filenames['rules'],
                   ((str(rule), rule_freq[str(rule)], domsize)\
                    for rule, domsize in \
                        compute_rule_domsizes(lexicon_tr, rules)))


def run_bipartite() -> None:
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon(filename=shared.filenames['wordlist'])
    wordlist_left = load_normalized_wordlist(
                        shared.filenames['wordlist.left'])
    wordlist_right = load_normalized_wordlist(
                         shared.filenames['wordlist.right'])
    logging.getLogger('main').info('Building the lexicon transducers...')
    compile_lexicon_transducer(list(lexicon[word] for word in wordlist_left),
                               shared.filenames['left-tr'])
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

def run() -> None:
    if file_exists(shared.filenames['wordlist.left']) and\
       file_exists(shared.filenames['wordlist.right']):
        run_bipartite()
    elif file_exists(shared.filenames['wordlist']):
        run_standard()
    else:
        raise RuntimeError('No input file supplied!')


def cleanup() -> None:
    remove_file_if_exists(shared.filenames['rules'])
    remove_file_if_exists(shared.filenames['lexicon-tr'])
    remove_file_if_exists(shared.filenames['graph'])

