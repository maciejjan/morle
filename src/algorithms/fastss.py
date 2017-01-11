from utils.files import *
from utils.printer import *
from collections import defaultdict
import re

# TODO remove the global variables and replace them with settings
MAX_DIST = 10
MIN_LENGTH = 2
MIN_COMP_LEN = 5
MAX_INFIX_LENGTH = 3
MAX_AFFIX_LENGTH = 5
INFIX_SLOTS = 1
MAX_OUTPUT_SIZE = 5e10    # 50 GB

# TODO compounds

def slice_word(word, slices):
    results = []
    queue = [(word, slices)]
    visited = set()
    while queue:
        word, slices = queue.pop()
        if slices in visited: continue
        visited.add(slices)
        substr = sum((word[i:j] for i, j in slices), ())
        results.append(''.join(substr))
        if len(substr) <= len(word)/2: continue
        for i, (begin, end) in enumerate(slices):
            if begin < end-1:
                if i == 0:
                    if begin < MAX_AFFIX_LENGTH:
                        queue.append((word, ((begin+1, end),) + slices[1:]))
                elif begin-slices[i-1][1]+1 < MAX_INFIX_LENGTH:
                    queue.append(\
                        (word,
                         slices[:i] + ((begin+1, end),) + slices[i+1:]))
                if i == len(slices)-1:
                    if len(word)-end+1 < MAX_AFFIX_LENGTH:
                        queue.append((word, slices[:i] + ((begin, end-1),)))
                elif slices[i+1][0]-end+1 < MAX_INFIX_LENGTH:
                    queue.append(\
                        (word,
                         slices[:i] + ((begin, end-1),) + slices[i+1:]))
            if len(slices) < INFIX_SLOTS + 1:
                for mid in range(begin+1, end-1):
                    queue.append((\
                        word,
                        slices[:i] +\
                            ((begin, mid), (mid+1, end)) +\
                            slices[i+1:]))
    return results

def substrings_for_word(word):
    return set(slice_word(word, ((0, len(word)),)))

def generate_substrings(lexicon, print_progress=False):
    pp = progress_printer(len(lexicon)) if print_progress else None
    for node in lexicon.values():
        for substr in substrings_for_word(node.word):
            yield (substr, node)
        if print_progress:
            next(pp)

def create_substrings_file(lexicon, output_file):
    with open_to_write(output_file) as fp:
        for substr, node in generate_substrings(lexicon, print_progress=True):
            write_line(fp, (substr, ''.join(node.word)))

def create_substrings_hash(lexicon):
    result = defaultdict(lambda: list())
    for substr, node in generate_substrings(lexicon, print_progress=True):
        result[substr].append(node)
    return result

def similar_words(lexicon, print_progress=False):
    substr_hash = defaultdict(lambda: list())
    if print_progress:
        pp = progress_printer(len(lexicon))
    for node in lexicon.values():
        sim_nodes = set()
        for substr in substrings_for_word(node.word):
            for node2 in substr_hash[substr]:
                sim_nodes.add(node2)
            substr_hash[substr].append(node)
        for node2 in sim_nodes:
            yield node, node2
        if print_progress:
            next(pp)

def similar_words_bipartite(lexicon_left, lexicon_right, print_progress=False):
    substr_hash = defaultdict(lambda: list())
    for node in lexicon_left.values():
        for substr in substrings_for_word(node.word):
            substr_hash[substr].append(node)
    if print_progress:
        pp = progress_printer(len(lexicon_right))
    for node in lexicon_right.values():
        sim_nodes = set()
        for substr in substrings_for_word(node.word):
            for node2 in substr_hash[substr]:
                sim_nodes.add(node2)
        for node2 in sim_nodes:
            yield node2, node
        if print_progress:
            next(pp)

