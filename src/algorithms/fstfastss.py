from algorithms.fst import binary_disjunct, delenv, delfilter, seq_to_transducer
from utils.printer import progress_printer
import shared

from collections import defaultdict
import hfst
import subprocess
import os.path

def build_substring_transducer(lexicon_tr, max_word_len):
    alphabet = lexicon_tr.get_alphabet()
    d = delenv(alphabet,
               shared.config['preprocess'].getint('max_affix_length'),
               shared.config['preprocess'].getint('max_infix_length'),
               shared.config['preprocess'].getint('max_infix_slots'))
    d.compose(delfilter(alphabet, max_word_len))
    d.minimize()
    
    result = hfst.HfstTransducer(lexicon_tr)
    result.compose(d)
    result.minimize()
    return result

def similar_words(tr_left, tr_right=None):

#     def remove_eps(sequence):
#         return tuple(symbol for symbol in sequence 
#                             if symbol and symbol != hfst.EPSILON)
    
    input_words = list(tr_left.extract_paths())
    # TODO max_word_len - more exactly!!! different for right and left,
    #      take multi-character symbols into account
#     max_word_len = max(len(word) for word in input_words)
# 
#     substr_tr_left = build_substring_transducer(tr_left, max_word_len)
#     if tr_right is None:
#         substr_tr_right = hfst.HfstTransducer(substr_tr_left)
#     else:
#         substr_tr_right = build_substring_transducer(tr_right, max_word_len)
#     substr_tr_right.invert()
# 
#     substr_tr_left.convert(hfst.ImplementationType.HFST_OL_TYPE)
#     substr_tr_right.convert(hfst.ImplementationType.HFST_OL_TYPE)
# 
    tr_path = os.path.join(shared.options['working_dir'], 'fastss.fsm')
#     ostr = hfst.HfstOutputStream(filename=tr_path, 
#                                  type=hfst.ImplementationType.HFST_OL_TYPE)
#     try:
#         ostr.write(substr_tr_left)
#         ostr.write(substr_tr_right)
#         ostr.flush()
#     except Exception:
#         pass
#     finally:
#         ostr.close()

    cmd = ['hfst-lookup', '-i', tr_path, '-C', 'composition']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, universal_newlines=True)
    try:
        pout, perr = p.communicate('\n'.join(input_words[:100]))
        for line in pout.split('\n'):
            yield line
    except Exception:
        pass
    finally:
        p.stdin.close()
        p.wait()

#     words_left = list(tr_left.extract_paths())
#     max_word_len = max(len(word) for word in words_left)
# #     words_right = list(tr_right.extract_paths()) if tr_right is not None\
# #                                                  else words_left
#     my_tr_left = hfst.HfstTransducer(tr_left)
#     d = delenv(tr_left.get_alphabet(),
#                shared.config['preprocess'].getint('max_affix_length'),
#                shared.config['preprocess'].getint('max_infix_length'),
#                shared.config['preprocess'].getint('max_infix_slots'))
#     d.compose(delfilter(tr_left.get_alphabet(), max_word_len))
#     d.minimize()
#     my_tr_left.compose(d)
# #     my_tr_left.remove_epsilons()
#     my_tr_left.minimize()
#     if tr_right is None:
#         my_tr_right = hfst.HfstTransducer(my_tr_left)
#     else:
#         my_tr_right = hfst.HfstTransducer(tr_right)
#         my_tr_right.compose(
#             delenv(tr_right.get_alphabet(),
#                    shared.config['preprocess'].getint('max_affix_length'),
#                    shared.config['preprocess'].getint('max_infix_length'),
#                    shared.config['preprocess'].getint('max_infix_slots')))
#         my_tr_right.minimize()
#     my_tr_right.invert()
#     my_tr_left.convert(hfst.ImplementationType.HFST_OL_TYPE)
#     my_tr_right.convert(hfst.ImplementationType.HFST_OL_TYPE)
# #     my_tr_left.convert(hfst.ImplementationType.SFST_TYPE)
# #     my_tr_right.convert(hfst.ImplementationType.SFST_TYPE)
#     return my_tr_left, my_tr_right

#     count = 0
#     for word_left in words_left:
#         words = set([word for substr, cost in my_tr_left.lookup(word_left)\
#                           for word, cost in my_tr_right.lookup(substr)])
#         for word_right in words:
#             yield (word_left, word_right)
#         count += 1
#         print('\r', count, sep='', end='')

