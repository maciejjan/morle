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


def tokenize_word(string :str) -> Tuple[List[str], List[str], str]:
    '''Separate a string into a word and a POS-tag,
       both expressed as sequences of symbols.'''
    pat_word = shared.compiled_patterns['word']
    pat_symbol = shared.compiled_patterns['symbol']
    pat_tag = shared.compiled_patterns['tag']

    m = re.match(pat_word, string)
    if m is None:
        raise Exception('Error while tokenizing word: %s' % string)
    return tuple(re.findall(pat_symbol, m.group('word'))),\
           tuple(re.findall(pat_tag, m.group('tag'))),\
           m.group('disamb')


def normalize_seq(seq :List[str]) -> List[str]:

    def _is_uppercase_letter(c :str) -> bool:
        return c.isupper() and not c in shared.multichar_symbols

    def _normalize_symbol(c :str) -> Iterable[str]:
        if _is_uppercase_letter(c):
            return ('{CAP}', c.lower())
        elif c in shared.normalization_substitutions:
            return (shared.normalization_substitutions[c],)
        else:
            return (c,)

    if all(_is_uppercase_letter(c) for c in seq):
        return ('{ALLCAPS}',) + tuple(c.lower() for c in seq)
    else:
        return tuple(itertools.chain.from_iterable(
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
    return tuple(result)


def normalize_word(literal :str) -> str:
    word, tag, disamb = tokenize_word(literal)
    return ''.join(normalize_seq(word) + tag)


def unnormalize_word(literal :str) -> str:
    word, tag, disamb = tokenize_word(literal)
    return ''.join(unnormalize_seq(word) + tag)


class LexiconEntry:

    def __init__(self, word, **kwargs) -> None:
        # read arguments one by one
        self.literal = word
        self.word, self.tag, self.disamb = tokenize_word(self.literal)
        self.word = normalize_seq(self.word)
        self.symstr = ''.join(self.word + self.tag)
        self.normalized = ''.join(self.word + self.tag) +\
                          ((shared.format['word_disamb_sep'] + self.disamb) \
                           if self.disamb is not None else '')
        self._is_possible_edge_source = \
            kwargs['is_possible_edge_source'] \
            if 'is_possible_edge_source' in kwargs else True
        self._is_possible_edge_target = \
            kwargs['is_possible_edge_target'] \
            if 'is_possible_edge_target' in kwargs else True
        if 'freq' in kwargs:
            self.freq = kwargs['freq']
            self.logfreq = math.log(self.freq)
        if 'vec' in kwargs:
            self.vec = kwargs['vec']

    def copy(self) -> 'LexiconEntry':
        kwargs = {
            'is_possible_edge_source' : self.is_possible_edge_source,
            'is_possible_edge_target' : self.is_possible_edge_target
        }
        if hasattr(self, 'vec'):
            kwargs['vec'] = self.vec
        result = LexiconEntry(self.literal, **kwargs)
        return result

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
    def __init__(self,
                 items :Union[LexiconEntry, Iterable[LexiconEntry]] = None) \
                -> None:
        self.items = []           # type: List[LexiconEntry]
        self.index = {}           # type: Dict[LexiconEntry, int]
        self.items_by_key = {}    # type: Dict[str, LexiconEntry]
        self.items_by_symstr = {} # type: Dict[str, List[LexiconEntry]]
        self.next_id = 0
        self.alphabet = set()
        self.tagset = set()
        self.max_word_length = 0
        self.max_symstr_length = 0
        if shared.config['Models'].get('root_feature_model') != 'none':
            dim = shared.config['Features'].getint('word_vec_dim')
#             self.feature_matrix = np.ndarray((0, dim))
        if items:
            self.add(items)

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

    def get_alphabet(self) -> List[str]:
        return sorted(list(self.alphabet))

    def get_tagset(self) -> List[Iterable[str]]:
        return sorted(list(self.tagset))

    def get_max_word_length(self) -> int:
        return self.max_word_length

    def get_max_symstr_length(self) -> int:
        return self.max_symstr_length

    def symstrs(self) -> Iterable[str]:
        return self.items_by_symstr.keys()

    def add(self, items :Union[LexiconEntry, Iterable[LexiconEntry]]) -> None:
        if isinstance(items, LexiconEntry):
            items = [items]
#         if not isinstance(items, list):
#             items = list(items)
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
            self.alphabet |= set(item.word + item.tag)
            self.tagset.add(item.tag)
            self.max_word_length = max(self.max_word_length, len(item.word))
            self.max_symstr_length = max(self.max_symstr_length,
                                       len(item.word) + len(item.tag))
#         if shared.config['Models'].get('root_feature_model') != 'none':
#             self.feature_matrix = \
#                 np.vstack((self.feature_matrix,
#                            np.array([item.vec for item in items])))


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

        def _parse_entry_from_row(row :List[str], use_restr=False,
                                  use_freq=False, use_vec=False, vec_sep=' ',
                                  vec_dim=None)\
                                 -> LexiconEntry:
            my_row = list(row)      # copy because it will be destroyed
            word = my_row.pop(0)
            kwargs = {}
            if use_restr:
                restr = my_row.pop(0).strip()
                kwargs['is_possible_edge_source'] = 'L' in restr
                kwargs['is_possible_edge_target'] = 'R' in restr
            if use_freq:
                kwargs['freq'] = int(my_row.pop(0).strip())
            if use_vec:
                vec_str = my_row.pop(0).strip()
                kwargs['vec'] = \
                    np.array(list(map(float, vec_str.split(vec_sep))))
                if kwargs['vec'] is None:
                    raise Exception("%s vec=None" % word)
                if kwargs['vec'].shape[0] != vec_dim:
                    raise Exception("%s dim=%d" % \
                                    (word, kwargs['vec'].shape[0]))
            return LexiconEntry(word, **kwargs)

        lexicon = Lexicon()
        # determine the file format
        use_restr = \
            shared.config['General'].getboolean('use_edge_restrictions')
        use_freq = \
            shared.config['General'].getboolean('use_frequency')
        use_vec = \
            shared.config['Models'].get('root_feature_model') != 'none' or \
            shared.config['Models'].get('edge_feature_model') != 'none'
        supervised = shared.config['General'].getboolean('supervised')
        vec_sep = shared.format['vector_sep']
        vec_dim = shared.config['Features'].getint('word_vec_dim')
        kwargs = { 'use_restr' : use_restr, 'use_freq' : use_freq, 
                   'use_vec' : use_vec, 'vec_dim' : vec_dim }
        items_to_add = []
        for row in read_tsv_file(filename):
            try:
                if supervised:
                    row.pop(0)    # the first item is the base/lemma -> ignore
                entry = _parse_entry_from_row(row, **kwargs)
                items_to_add.append(entry)
            except Exception as e:
#                 raise e
                logging.getLogger('main').warning('ignoring %s: %s' %\
                                                  (row[0], str(e)))
        lexicon.add(items_to_add)
        return lexicon

    def _lexc_escape(self, string :str) -> str:
        '''Escape a string for correct rendering in a LEXC file.'''
        return re.sub('([0<>])', '%\\1', string)


def load_raw_vocabulary(filename :str) -> Lexicon:
    lexicon = Lexicon()
    for (word,) in read_tsv_file(filename):
        try:
            lexicon.add(LexiconEntry(word))
        except Exception as e:
            logging.getLogger('main').warning('ignoring %s: %s' %\
                                              (word, str(e)))
    return lexicon

