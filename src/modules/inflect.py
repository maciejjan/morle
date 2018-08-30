import algorithms.fst
from datastruct.lexicon import tokenize_word, normalize_seq, normalize_word, \
                               unnormalize_word
from utils.files import read_tsv_file
import shared

import hfst
import tqdm


def run():
    lexicon_tr = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    alphabet = lexicon_tr.get_alphabet()
    pairs = [(lemma, tag) \
               for lemma, tag in \
                   read_tsv_file(shared.filenames['analyze.wordlist'])]
    for lemma, tag in tqdm.tqdm(pairs):
        try:
            lemma_seq, lemma_tag_seq, lemma_disamb = tokenize_word(lemma)
            lt_seq = normalize_seq(lemma_seq) + lemma_tag_seq
            tag_seq = tokenize_word('z'+tag)[1]
            tr = algorithms.fst.seq_to_transducer(list(zip(lt_seq, lt_seq)))
            tr.compose(rules_tr)
            tr.minimize()
            tr.compose(algorithms.fst.tag_acceptor(tag_seq, alphabet))
            tr.minimize()
            tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
            lookup_results = tr.lookup(normalize_word(lemma), max_number=1)
            if lookup_results:
                word_str = unnormalize_word(lookup_results[0][0])
                print(lemma, word_str, sep='\t')
        except Exception:
            pass

