from utils.files import full_path, open_to_write, read_tsv_file, remove_file
import shared

from collections import defaultdict
import hfst
import itertools
import numpy as np
import re
import math
import logging
from typing import Any, Dict, Callable, Iterable, List, Tuple, Union


# def get_wordlist_format() -> List[Callable[[Any], Any]]:
#     result = [str]  # type: List[Callable[[Any], Any]]
#     if shared.config['General'].getboolean('supervised'):
#         result.append(str)
#     if shared.config['Features'].getfloat('word_freq_weight') > 0:
#         result.append(int)
#     if shared.config['Features'].getfloat('word_vec_weight') > 0:
#         vector_sep = shared.format['vector_sep']
#         result.append(\
#             lambda x: np.array(list(map(float, x.split(vector_sep)))))
#     return result


def tokenize_word(string :str) -> Tuple[List[str], List[str], str]:
    '''Separate a string into a word and a POS-tag,
       both expressed as sequences of symbols.'''
    pat_word = shared.compiled_patterns['word']
    pat_symbol = shared.compiled_patterns['symbol']
    pat_tag = shared.compiled_patterns['tag']

    m = re.match(pat_word, string)
    if m is None:
        raise Exception('Error while tokenizing word: %s' % string)
    return list(re.findall(pat_symbol, m.group('word'))),\
           list(re.findall(pat_tag, m.group('tag'))),\
           m.group('disamb')


def normalize_seq(seq :List[str]) -> List[str]:

    def _is_uppercase_letter(c :str) -> bool:
        return c.isupper() and not c in shared.multichar_symbols

    def _normalize_symbol(c :str) -> Iterable[str]:
        if _is_uppercase_letter(c):
            return ['{CAP}', c.lower()]
        elif c in shared.normalization_substitutions:
            return [shared.normalization_substitutions[c]]
        else:
            return [c]

    if all(_is_uppercase_letter(c) for c in seq):
        return ['{ALLCAPS}'] + [c.lower() for c in seq]
    else:
        return list(itertools.chain.from_iterable(
                        _normalize_symbol(c) for c in seq))


def unnormalize_seq(seq :List[str]) -> List[str]:
    result = []
    allcaps = False
    cap = False
    for c in seq:
        if c == '{ALLCAPS}':
            allcaps = True
        elif c == '{CAP}':
            cap = True
        else:
            if c in shared.unnormalization_substitutions:
                result.append(shared.unnormalization_substitutions[c])
            elif allcaps or cap:
                result.append(c.upper())
            else:
                result.append(c)
            cap = False
    return result


def normalize_word(literal :str) -> str:
    word, tag, disamb = tokenize_word(literal)
    return ''.join(normalize_seq(word) + tag)


def unnormalize_word(literal :str) -> str:
    word, tag, disamb = tokenize_word(literal)
    return ''.join(unnormalize_seq(word) + tag)


class LexiconEntry:

    # TODO do not retrieve config for every lexicon entry!
    #      (strings have to be converted each time)
    def __init__(self, *args) -> None:
        # read arguments one by one
        args = list(args)
        self.literal = args.pop(0)
        self.word, self.tag, self.disamb = tokenize_word(self.literal)
        self.word = normalize_seq(self.word)
        # string of printable symbols -- does not include disambiguation IDs
        self.symstr = ''.join(self.word + self.tag)
        self.normalized = ''.join(self.word + self.tag) +\
                          ((shared.format['word_disamb_sep'] + self.disamb) \
                           if self.disamb is not None else '')
        if shared.config['General'].getboolean('use_edge_restrictions'):
            edge_restrictions = args.pop(0)
            self._is_possible_edge_source = ('L' in edge_restrictions)
            self._is_possible_edge_target = ('R' in edge_restrictions)
        else:
            self._is_possible_edge_source = True
            self._is_possible_edge_target = True
        if shared.config['Features'].getfloat('word_freq_weight') > 0:
            self.freq = int(args.pop(0))
            self.logfreq = math.log(self.freq)
        if shared.config['Features'].getfloat('word_vec_weight') > 0:
            vec_sep = shared.format['vector_sep']
            self.vec = np.array(list(map(float, args.pop(0).split(vec_sep))))
            if self.vec is None:
                raise Exception("%s vec=None" % (self.literal))
            if self.vec.shape[0] != shared.config['Features']\
                                          .getfloat('word_vec_dim'):
                raise Exception("%s dim=%d" %\
                                (self.literal, self.vec.shape[0]))

    def __lt__(self, other) -> bool:
        if not isinstance(other, LexiconEntry):
            raise TypeError('Expected LexiconEntry, got %s', type(other))
        return self.literal < other.literal
    
    def __eq__(self, other) -> bool:
        return isinstance(other, LexiconEntry) and \
               self.literal == other.literal

    def __str__(self) -> str:
        return self.literal

    def __hash__(self) -> int:
        return self.literal.__hash__()

    def is_possible_edge_source(self) -> bool:
        return self._is_possible_edge_source

    def is_possible_edge_target(self) -> bool:
        return self._is_possible_edge_target
    
    def to_fst(self) -> hfst.HfstTransducer:
        return hfst.fst(self.symstr)
    

class Lexicon:
    def __init__(self) -> None:
        self.items = []           # type: List[LexiconEntry]
        self.index = {}           # type: Dict[LexiconEntry, int]
        self.items_by_key = {}    # type: Dict[str, LexiconEntry]
        self.items_by_symstr = {} # type: Dict[str, List[LexiconEntry]]
        self.next_id = 0
        if shared.config['Models'].get('root_feature_model') != 'none':
            dim = shared.config['Features'].getint('word_vec_dim')
            self.feature_matrix = np.ndarray((0, dim))

    def __contains__(self, key :Union[str, LexiconEntry]) -> bool:
        if isinstance(key, LexiconEntry):
            return key in self.index
        return key in self.items_by_key

    def __getitem__(self, key :Union[int, str]) -> LexiconEntry:
        if isinstance(key, str):
            return self.items_by_key[key]
        elif isinstance(key, int):
            return self.items[key]
        else:
            raise KeyError(key)

    def get_by_id(self, idx :int) -> LexiconEntry:
        return self.items[idx]

    def get_id(self, entry :LexiconEntry) -> int:
        return self.index[entry]

    def get_by_symstr(self, key :str) -> List[LexiconEntry]:
        return self.items_by_symstr[key]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterable[LexiconEntry]:
        return iter(self.items)

    def keys(self) -> Iterable[str]:
        return self.items_by_key.keys()

#     def get_alphabet(self) -> Tuple[str]:
#         alphabet = set()
#         for item in self:
#             alphabet |= set(item.word)
#             alphabet |= set(item.tag)
#         return tuple(sorted(list(alphabet)))

    def symstrs(self) -> Iterable[str]:
        return self.items_by_symstr.keys()

    def add(self, items :Union[LexiconEntry, Iterable[LexiconEntry]]) -> None:
        if isinstance(items, LexiconEntry):
            items = [items]
        if not isinstance(items, list):
            items = list(items)
        for item in items:
            if str(item) in self.items_by_key:
                raise ValueError('{} already in vocabulary'.format(str(item)))
            if not item.symstr in self.items_by_symstr:
                self.items_by_symstr[item.symstr] = []
            self.items.append(item)
            self.index[item] = self.next_id
            self.items_by_key[str(item)] = item
            self.items_by_symstr[item.symstr].append(item)
            self.next_id += 1
        if shared.config['Models'].get('root_feature_model') != 'none':
            self.feature_matrix = \
                np.vstack((self.feature_matrix,
                           np.array([item.vec for item in items])))

    def to_fst(self) -> hfst.HfstTransducer:
        lexc_file = shared.filenames['lexicon-tr'] + '.lex'
        tags = set()
        for entry in self.items:
            for t in entry.tag:
                tags.add(t)
        with open_to_write(lexc_file) as lexfp:
            lexfp.write('Multichar_Symbols ' + 
                        ' '.join(self._lexc_escape(s) \
                        for s in shared.multichar_symbols+list(tags)) + '\n\n')
            lexfp.write('LEXICON Root\n')
            for entry in self.items:
                lexfp.write('\t' + self._lexc_escape(entry.symstr) + ' # ;\n')
        transducer = hfst.compile_lexc_file(full_path(lexc_file))
        remove_file(lexc_file)
        return transducer

    def remove(self, item :LexiconEntry) -> None:
        if str(item) not in self.items:
            raise KeyError(str(item))
        del self.items[str(item)]
        if item.symstr in self.items_by_symstr:
            self.items_by_symstr[item.symstr].remove(item)
            if not self.items_by_symstr[item.symstr]:
                del self.items_by_symstr[item.symstr]

    @staticmethod
    def load(filename :str) -> 'Lexicon':
        lexicon = Lexicon()
        items_to_add = []
        for row in read_tsv_file(filename):
            try:
                items_to_add.append(LexiconEntry(*row))
            except Exception as e:
#                 raise e
                logging.getLogger('main').warning('ignoring %s: %s' %\
                                                  (row[0], str(e)))
        lexicon.add(items_to_add)
        return lexicon

    def _lexc_escape(self, string :str) -> str:
        '''Escape a string for correct rendering in a LEXC file.'''
        return re.sub('([0<>])', '%\\1', string)

    def _lexc_escape(self, string :str) -> str:
        '''Escape a string for correct rendering in a LEXC file.'''
        return re.sub('([0<>])', '%\\1', string)


