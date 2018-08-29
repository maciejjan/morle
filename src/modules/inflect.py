import algorithms.fst
from datastruct.lexicon import tokenize_word
from utils.files import read_tsv_file
import shared

import tqdm


def run():
    lexicon_tr = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    alphabet = lexicon_tr.get_alphabet()
    for lemma, tag in read_tsv_file(shared.filenames['analyze.wordlist']):
        lemma_seq, lemma_tag_seq, lemma_disamb = tokenize_word(lemma)
        lt_seq = lemma_seq + lemma_tag_seq
        tag_seq = tokenize_word('z'+tag)[1]
        tr = algorithms.fst.seq_to_transducer(list(zip(lt_seq, lt_seq)))
        tr.compose(rules_tr)
        tr.minimize()
        tr.compose(algorithms.fst.tag_acceptor(tag_seq, alphabet))
        tr.minimize()
        print(tr.n_best(1), tr.extract_paths())
    # TODO input format:
    # base<tag> tab <tag>
    # output format:
    # base<tag> tab word<tag>
    # TODO flow:
    # - create a transducer: base .o. rules .o. (universal_letter_acceptor . tag_acceptor(tag))
    # - lookup the best path in the transducer
    pass

