import algorithms.align
import algorithms.fastss
import algorithms.fst
import algorithms.splrules
from datastruct.lexicon import *
from datastruct.rules import *
from models.point import *
from utils.files import *
from utils.printer import *
import logging

def expand_graph(graph_file):
    '''Annotate graph with additional information needed for filtering:
       currently rule frequencies.'''
    with open_to_write(graph_file + '.tmp') as graph_tmp_fp:
        for rule, wordpairs in read_tsv_file_by_key(graph_file, 3, 
                print_progress=True, print_msg='Expanding the graph for filtering...'):
            freq = len(wordpairs)
            for w1, w2 in wordpairs:
                write_line(graph_tmp_fp, (w1, w2, rule, freq))
    rename_file(graph_file + '.tmp', graph_file)

def contract_graph(graph_file):
    '''Remove any additional information needed for filtering.'''
    with open_to_write(graph_file + '.tmp') as graph_tmp_fp:
        for w1, w2, rule, freq in read_tsv_file(graph_file,
                print_progress=True, print_msg='Contracting the graph...'):
            write_line(graph_tmp_fp, (w1, w2, rule))
    rename_file(graph_file + '.tmp', graph_file)

def filter_max_num_rules(graph_file):
    logging.getLogger('main').info('filter_max_num_rules')
    sort_file(graph_file, stable=True, numeric=True, reverse=True, key=4)
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
    sort_file(graph_file, stable=True, key=(1, 2))
    with open_to_write(graph_file + '.tmp') as graph_fil_fp:
        for (word_1, word_2), edges in read_tsv_file_by_key(graph_file, (1, 2),
                print_progress=True, print_msg='filter_max_edges_per_wordpair'):
            for rule, freq in edges[:shared.config['preprocess'].getint('max_edges_per_wordpair')]:
                write_line(graph_fil_fp, (word_1, word_2, rule, freq))
    rename_file(graph_file + '.tmp', graph_file)
    sort_file(graph_file, key=3)
    sort_file(graph_file, stable=True, numeric=True, reverse=True, key=4)
    update_file_size(graph_file)

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
    sort_file(graph_file + '.filtered', key=3)
    sort_file(graph_file + '.filtered', stable=True, numeric=True, reverse=True, key=4)
    remove_file(graph_file)
    remove_file(graph_file + '.tmp2')
    # cleanup files
#    rename_file(graph_file, graph_file + '.orig')
    rename_file(graph_file + '.filtered', graph_file)


def build_graph(lexicon, graph_file):
    with open_to_write(graph_file) as fp:
        for n1, n2 in algorithms.fastss.similar_words(lexicon, print_progress=True):
            rule = algorithms.align.extract_rule(n1, n2)
            if rule is not None:
                write_line(fp, (str(n1), str(n2), str(rule)))
                write_line(fp, (str(n2), str(n1), rule.reverse().to_string()))

def build_graph_allrules(lexicon, graph_file):
    with open_to_write(graph_file) as fp:
        for n1, n2 in algorithms.fastss.similar_words(lexicon, print_progress=True):
            for rule in algorithms.align.extract_all_rules(n1, n2):
                write_line(fp, (str(n1), str(n2), str(rule)))
                write_line(fp, (str(n2), str(n1), rule.reverse().to_string()))

def build_graph_allrules_fil(lexicon, graph_file, filters):
    with open_to_write(graph_file) as fp:
        for n1, n2 in algorithms.fastss.similar_words(lexicon, print_progress=True):
            for rule in algorithms.align.extract_all_rules(n1, n2):
                if all(f.check(n1, n2) for f in filters):
                    write_line(fp, (str(n1), str(n2), str(rule)))
                if all(f.check(n2, n1) for f in filters):
                    write_line(fp, (str(n2), str(n1), rule.reverse().to_string()))

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

def compute_rule_domsizes(lexicon, rules_file):
    with open_to_write(rules_file + '.tmp') as outfp:
        for rule_str, freq in read_tsv_file(rules_file, (str, int), print_progress=True,\
                print_msg='Estimating rule domain sizes...'):
            rule = Rule.from_string(rule_str)
            domsize = rule.compute_domsize(lexicon)
            write_line(outfp, (rule, freq, domsize))
    rename_file(rules_file + '.tmp', rules_file)

def split_rules_in_graph(lexicon, graph_file, model):
    with open_to_write(graph_file + '.spl') as outfp:
        for rule_str, wordpairs in read_tsv_file_by_key(graph_file, key=3,\
                print_progress=True):
            rule = Rule.from_string(rule_str)
            edges = [LexiconEdge(lexicon[w1], lexicon[w2], rule)\
                for w1, w2 in wordpairs]
            edges_spl =\
                algorithms.splrules.split_rule(rule_str, edges, lexicon, model)
            for e in edges_spl:
                write_line(outfp,\
                    (str(e.source), str(e.target), str(e.rule)))
    rename_file(graph_file + '.spl', graph_file)

### MAIN FUNCTIONS ###

def run():
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])

    if shared.config['General'].getboolean('supervised'):
        logging.getLogger('main').info('Building graph...')
        build_graph_from_training_edges(lexicon,
                                        shared.filenames['wordlist'],
                                        shared.filenames['graph'])
    else:
        logging.getLogger('main').info('Building graph...')
        build_graph_allrules(lexicon, shared.filenames['graph'])

    sort_file(shared.filenames['graph'], key=3)
    update_file_size(shared.filenames['graph'])
    run_filters(shared.filenames['graph'])
    update_file_size(shared.filenames['graph'])
    aggregate_file(shared.filenames['graph'],\
                   shared.filenames['rules'], 3)
    update_file_size(shared.filenames['rules'])

    lexicon.build_transducer()
    algorithms.fst.save_transducer(lexicon.transducer,
                                   shared.filenames['lexicon-tr'])
    compute_rule_domsizes(lexicon, shared.filenames['rules'])

def cleanup():
    remove_file_if_exists(shared.filenames['rules'])
    remove_file_if_exists(shared.filenames['lexicon-tr'])
    remove_file_if_exists(shared.filenames['graph'])

def evaluate():
    print('\nSurface rules: nothing to evaluate.\n')

def import_from_db():
    utils.db.connect()
    print('Importing wordlist...')
    utils.db.pull_table(settings.WORDS_TABLE, ('word', 'freq'),\
        settings.FILES['training.wordlist'])
    print('Importing surface rules...')
    utils.db.pull_table(settings.S_RUL_TABLE, ('rule', 'freq', 'prob'),\
        settings.FILES['surface.rules'])
    # pull graph
    print('Importing graph...')
    with open_to_write(settings.FILES['surface.graph']) as fp:
        for word_1, word_2, rule in utils.db.query_fetch_results('''
            SELECT w1.word, w2.word, r.rule FROM graph g 
                JOIN words w1 ON g.w1_id = w1.w_id
                JOIN words w2 ON g.w2_id = w2.w_id
                JOIN s_rul r ON g.r_id = r.r_id
            ;'''):
            write_line(fp, (word_1, word_2, rule))
    # pull surface rules co-occurrences
    print('Importing surface rules co-occurrences...')
    with open_to_write(settings.FILES['surface.rules.cooc']) as fp:
        for rule_1, rule_2, freq, sig in utils.db.query_fetch_results('''
            SELECT r1.rule, r2.rule, c.freq, c.sig FROM s_rul_co c
                JOIN s_rul r1 ON c.r1_id = r1.r_id
                JOIN s_rul r2 ON c.r2_id = r2.r_id
            ;'''):
            write_line(fp, (rule_1, rule_2, freq, sig))
    utils.db.close_connection()

def export_to_db():
    # words <- insert ID
    print('Converting wordlist...')
    word_ids = utils.db.insert_id(settings.FILES['training.wordlist'],\
        settings.FILES['wordlist.db'])
    # surface rules <- insert ID
    print('Converting surface rules...')
    s_rule_ids = utils.db.insert_id(settings.FILES['surface.rules'],\
        settings.FILES['surface.rules.db'])
    # graph <- replace words and surface rules with their ID
    print('Converting graph...')
    utils.db.replace_values_with_ids(settings.FILES['surface.graph'],\
        settings.FILES['surface.graph.db'],\
        (word_ids, word_ids, s_rule_ids))
    # surface_rules_cooc <- replace rules with ID
    print('Converting surface rules co-occurrences...')
    utils.db.replace_values_with_ids(settings.FILES['surface.rules.cooc'],\
        settings.FILES['surface.rules.cooc.db'],\
        (s_rule_ids, s_rule_ids, None, None))
    # load tables into DB
    utils.db.connect()
    print('Exporting wordlist...')
    utils.db.push_table(settings.WORDS_TABLE, settings.FILES['wordlist.db'])
    print('Exporting surface rules...')
    utils.db.push_table(settings.S_RUL_TABLE, settings.FILES['surface.rules.db'])
    print('Exporting graph...')
    utils.db.push_table(settings.GRAPH_TABLE, settings.FILES['surface.graph.db'])
    print('Exporting surface rules co-occurrences...')
    utils.db.push_table(settings.S_RUL_CO_TABLE,\
        settings.FILES['surface.rules.cooc.db'])
    utils.db.close_connection()
    # delete temporary files
    remove_file(settings.FILES['wordlist.db'])
    remove_file(settings.FILES['surface.rules.db'])
    remove_file(settings.FILES['surface.graph.db'])
    remove_file(settings.FILES['surface.rules.cooc.db'])

