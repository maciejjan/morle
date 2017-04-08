from datastruct.rules import *
import hfst
from operator import itemgetter


def extract_all_rules(node_1, node_2):
    '''Extract all rules fitting the pair of words.'''
    # queue: rule seq, remaining alignment, num_segments, length of the last segment, is_last_segment
    alignment = align_words(node_1.word, node_2.word)
    max_affix_length = shared.config['preprocess'].getint('max_affix_length')
    max_infix_length = shared.config['preprocess'].getint('max_infix_length')
    max_infix_slots = shared.config['preprocess'].getint('max_infix_slots')
#    print(alignment)
    queue = [((), alignment, 1, 0, False)]
    results = []
    while queue:
        seq, alignment, num_seg, len_seg, last_seg = queue.pop()
        if not alignment:
            results.append(Rule.from_seq(seq, (node_1.tag, node_2.tag)))
            continue
        x, y = alignment[0]
        if x == y:
            # prolong the prefix
            if num_seg == 1 and len_seg < max_affix_length:
                queue.append((seq + ((x, y),),
                              alignment[1:], num_seg, len_seg+1, last_seg))
            # prolong the suffix
            if num_seg > 1 and len_seg < max_affix_length:
                queue.append((seq + ((x, y),),
                              alignment[1:], num_seg, len_seg+1, True))
            if not last_seg:
                # prolong the current alternation
                if num_seg > 1 and not last_seg and len_seg < max_infix_length:
                    if seq[-1] != (hfst.IDENTITY, hfst.IDENTITY):
                        queue.append((seq + ((x, y),),
                                      alignment[1:], num_seg, len_seg+1, last_seg))
                # insert or continue an identity symbol
                if not seq or seq[-1] != (hfst.IDENTITY, hfst.IDENTITY):
                    if num_seg < max_infix_slots+2:
                        queue.append((seq + ((hfst.IDENTITY, hfst.IDENTITY),),
                                      alignment[1:], num_seg+1, 0, last_seg))
                else:
                    queue.append((seq, alignment[1:],\
                                  num_seg, len_seg, last_seg))
        else:
            queue.append((seq + ((x, y),), alignment[1:],\
                          num_seg, len_seg+1, last_seg))
    return list(set(results))

