from algorithms.fst import binary_disjunct, delenv, seq_to_transducer
from utils.printer import progress_printer
import shared

from collections import defaultdict
import hfst

def similar_words(tr_left, tr_right=None):

    def remove_eps(sequence):
        return tuple(symbol for symbol in sequence 
                            if symbol and symbol != hfst.EPSILON)

    words_left = list(tr_left.extract_paths())
#     words_right = list(tr_right.extract_paths()) if tr_right is not None\
#                                                  else words_left
    my_tr_left = hfst.HfstTransducer(tr_left)
    my_tr_left.compose(
        delenv(tr_left.get_alphabet(),
               shared.config['preprocess'].getint('max_affix_length'),
               shared.config['preprocess'].getint('max_infix_length'),
               shared.config['preprocess'].getint('max_infix_slots')))
    my_tr_left.remove_epsilons()
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
#     my_tr_right.convert(hfst.ImplementationType.HFST_OL_TYPE)
#     return my_tr_left, my_tr_right

    for word_left in words_left:
        substrings = [remove_eps(substr)\
                      for cost, substr in my_tr_left.lookup(word_left,
                                                            output='raw')]
        max_len = max(len(ss) for ss in substrings)
        seqs = [list(zip(ss, ss)) for ss in substrings 
                                  if len(ss) >= max_len / 2]
        t = binary_disjunct(seq_to_transducer(seq, alphabet=my_tr_left.get_alphabet()) for seq in seqs)
        t.compose(my_tr_right)
        t.output_project()
        t.minimize()
        for word_right in t.extract_paths():
            yield (word_left, word_right)

#     print('substr_list...')
#     substr_list = []
#     max_count = len(words_right)
#     for idx, word in enumerate(words_right):
#         substrings = set(substr for substr, cost in my_tr_right.lookup(word)\
#                                 if len(substr) >= len(word) / 2)
#         for substr in substrings:
#             substr_list.append((idx, substr.__hash__()))
#         print('\r', idx, '/', max_count, sep='', end='', flush=True)
#     print()
#     print('done')

#     substr_hash = None
#     if hash_substrings:
#         substr_hash = defaultdict(lambda: list())
#         count, max_count = 0, len(words_right)
#         for idx, word_right in enumerate(words_right):
#             for substr, cost in my_tr_right.lookup(word_right):
#                 if len(substr) >= len(word_right) / 2:
# #                     print(word_right, substr)
#                     substr_hash[substr].append(idx)
#             count += 1
#             print('\r', count, '/', max_count, sep='', end='', flush=True)
#     print()
# 
#     count, max_count = 0, len(words_left)
#     for word_left in words_left:
#         results = set()
#         num_substr = 0
#         for substr, cost in my_tr_left.lookup(word_left):
#             if len(substr) >= len(word_left) / 2:
#                 num_substr += 1
#                 if hash_substrings:
#                     for word_right_idx in substr_hash[substr]:
#                         results.add(words_right[word_right_idx])
#                 else:
#                     for word_right, cost in my_tr_right.lookup(substr):
#                         if len(substr) >= len(word_right) / 2:
#                             results.add(word_right)
#         for word_right in results:
#             yield (word_left, word_right)
#         count += 1
#         print('\r', count, '/', max_count, ' ', num_substr, sep='', end='', flush=True)
#     print()

