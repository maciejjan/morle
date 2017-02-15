from algorithms.fst import binary_disjunct, delenv, delfilter, seq_to_transducer
from utils.printer import progress_printer
import shared

from collections import defaultdict
import hfst

def similar_words(tr_left, tr_right=None):

    def remove_eps(sequence):
        return tuple(symbol for symbol in sequence 
                            if symbol and symbol != hfst.EPSILON)

    words_left = list(tr_left.extract_paths())
    max_word_len = max(len(word) for word in words_left)
#     words_right = list(tr_right.extract_paths()) if tr_right is not None\
#                                                  else words_left
    my_tr_left = hfst.HfstTransducer(tr_left)
    d = delenv(tr_left.get_alphabet(),
               shared.config['preprocess'].getint('max_affix_length'),
               shared.config['preprocess'].getint('max_infix_length'),
               shared.config['preprocess'].getint('max_infix_slots'))
    d.compose(delfilter(tr_left.get_alphabet(), max_word_len))
    d.minimize()
    my_tr_left.compose(d)
#     my_tr_left.remove_epsilons()
    my_tr_left.minimize()
    if tr_right is None:
        my_tr_right = hfst.HfstTransducer(my_tr_left)
    else:
        my_tr_right = hfst.HfstTransducer(tr_right)
        my_tr_right.compose(
            delenv(tr_right.get_alphabet(),
                   shared.config['preprocess'].getint('max_affix_length'),
                   shared.config['preprocess'].getint('max_infix_length'),
                   shared.config['preprocess'].getint('max_infix_slots')))
        my_tr_right.minimize()
    my_tr_right.invert()
    my_tr_left.convert(hfst.ImplementationType.HFST_OL_TYPE)
    my_tr_right.convert(hfst.ImplementationType.HFST_OL_TYPE)
#     return my_tr_left, my_tr_right

    count = 0
    for word_left in words_left:
        words = set([word for substr, cost in my_tr_left.lookup(word_left)\
                          for word, cost in my_tr_right.lookup(substr)])
        for word_right in words:
            yield (word_left, word_right)
        count += 1
        print('\r', count, sep='', end='')

