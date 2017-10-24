import algorithms.fst
from utils.files import *
import hfst
import re
from operator import itemgetter


class Rule:
    def __init__(self, subst, tag_subst=None, string=None):
        self.subst = subst
        self.tag_subst = tag_subst
        if string is None:
            self.string = self.to_string()
        else:
            self.string = string
        
    def __le__(self, other):
        pattern = 'X'.join(''.join(s[0]) for s in self.subst) +\
            (''.join(self.tag_subst[0]) if self.tag_subst else '')
        return other.lmatch_string(pattern)

    def __lt__(self, other):
        return self.string < other.string
    
    def __eq__(self, other):
        return self.subst == other.subst and self.tag_subst == other.tag_subst
    
    def __str__(self):
        return self.string
    
    def __hash__(self):
        return self.string.__hash__()
    
    def copy(self):
        return Rule(self.subst, self.tag_subst)
    
    # TODO more meaningful name
    # TODO proposition: check_constraints()
    def check(self, max_affix_length, max_infix_length, infix_slots):
        if len(self.subst) > infix_slots+2:
            return False
        if max(len(self.subst[0][0]), len(self.subst[0][1]),\
                len(self.subst[-1][0]), len(self.subst[-1][1])) >\
                max_affix_length:
            return False
        if max((max(len(s[0]), len(s[1])) for s in self.subst[1:-1]),\
                default=0) > max_infix_length:
            return False
        if len(self.subst) > 2 and any(len(s[0]) == 0 for s in self.subst[1:-1]):
            return False
        return True
    
    def compute_domsize(self, lexicon_tr):
        # instead of computing T_L .o. T_r, compute:
        #   inv(T_L .o. T_r) = inv(T_r) .o. inv(T_L) = inv(T_r) .o. T_L
        t = self.to_fst()
        t.invert()
        t.compose(lexicon_tr)
        t.determinize()
        t.minimize()
        if t.is_cyclic():
            logging.getLogger('main').warning('cyclic transducer for %s' %\
                                              self.__str__())
        return algorithms.fst.number_of_paths(t)

    def seq(self):
        x_seq, y_seq = [], []
        x_tseq, y_tseq = [], []
        for x, y in self.subst:
            x_seq.extend(x + (hfst.EPSILON,)*(len(y)-len(x)))
            x_seq.append(hfst.IDENTITY)
            y_seq.extend((hfst.EPSILON,)*(len(x)-len(y)) + y)
            y_seq.append(hfst.IDENTITY)
        x_seq.pop()        # remove the last identity symbol
        y_seq.pop()
        if self.tag_subst:
            xt, yt = self.tag_subst
            x_seq.extend(xt + (hfst.EPSILON,)*(len(yt)-len(xt)))
            y_seq.extend((hfst.EPSILON,)*(len(xt)-len(yt)) + yt)
        return tuple(zip(x_seq, y_seq))
    
    def input_seq(self):
        seq = []
        for x, y in self.subst[:-1]:
            seq.extend(x)
            if not seq or seq[-1] != hfst.IDENTITY:
                seq.append(hfst.IDENTITY)
        seq.extend(self.subst[-1][0])
        if self.tag_subst:
            seq.extend(self.tag_subst[0])
        return tuple(seq)
    
    @staticmethod
    def from_seq(seq, tag_subst):
        subst = []
        x_seq, y_seq = (), ()
        for x, y in seq:
            if x == y == hfst.IDENTITY:
                subst.append((x_seq, y_seq))
                x_seq, y_seq = (), ()
            else:
                x_seq += (x,) if x != hfst.EPSILON else ()
                y_seq += (y,) if y != hfst.EPSILON else ()
        subst.append((x_seq, y_seq))
        tx, ty = tag_subst
        tag_subst = (tuple(tx), tuple(ty))
        return Rule(tuple(subst), tag_subst)
    
    def reverse(self):
        subst = tuple((y, x) for x, y in self.subst)
        tag_subst = self.tag_subst if not self.tag_subst \
            else (self.tag_subst[1], self.tag_subst[0])
        return Rule(subst, tag_subst)

    def to_string(self):
        return shared.format['rule_part_sep'].join(
            ''.join(x)+shared.format['rule_subst_sep']+''.join(y)
                for (x, y) in self.subst
            ) +\
            (shared.format['rule_tag_sep']+\
             ''.join(self.tag_subst[0]) +\
             shared.format['rule_subst_sep']+\
             ''.join(self.tag_subst[1]) if self.tag_subst else '')
    
    def to_fst(self, weight=0, alphabet=None):
        return algorithms.fst.seq_to_transducer(\
                   self.seq(), weight=weight, 
                   alphabet=shared.multichar_symbols)
    
    @staticmethod
    def from_string(string):
        r_subst, r_tag_subst = [], None
        m = re.match(shared.compiled_patterns['rule'], string)
        if m is None:
            raise Exception('Error while parsing rule: %s' % string)
        subst, tag_subst = m.group('subst'), m.group('tag_subst')
#        m_subst = re.match(settings.RULE_NAMED_SUBST_PATTERN_CMP, subst)
        for ms in re.finditer(shared.compiled_patterns['rule_named_subst'], subst):
            x = tuple(re.findall(shared.compiled_patterns['symbol'], ms.group('x')))
            y = tuple(re.findall(shared.compiled_patterns['symbol'], ms.group('y')))
            r_subst.append((x, y))
        if tag_subst:
            mt = re.match(shared.compiled_patterns['rule_named_tag_subst'], tag_subst)
            if mt is not None:
                x = tuple(re.findall(shared.compiled_patterns['tag'], mt.group('x')))
                y = tuple(re.findall(shared.compiled_patterns['tag'], mt.group('y')))
                r_tag_subst = (x, y)
        return Rule(tuple(r_subst), r_tag_subst, string=string)


class RuleSet:
    '''A class for saving/loading rules to/from a file and indexing
       (i.e. attributing IDs to) rules. Stores also domsizes.'''

    def __init__(self) -> None:
        self.items = []         # type: List[Rule]
        self.items_by_str = {}  # type: Dict[str, Rule]
        self.index = {}         # type: Dict[Rule, int]
        self.domsizes = {}      # type: Dict[Rule, domsize]
        self.next_id = 0

    def __contains__(self, rule :Rule) -> bool:
        return rule in self.index

    def __getitem__(self, key :Union[int, str]) -> Rule:
        if isinstance(key, str):
            return self.items_by_str[key]
        elif isinstance(key, int):
            return self.items[key]
        else:
            raise KeyError(key)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterable[Rule]:
        return iter(self.items)
    
    def add(self, rule :Rule, domsize :int) -> None:
        self.items.append(rule)
        self.index[rule] = self.next_id
        self.items_by_str[str(rule)] = rule
        self.domsizes[rule] = domsize
        self.next_id += 1

    def get_id(self, rule :Rule) -> int:
        return self.index[rule]

    def get_domsize(self, rule :Rule) -> int:
        return self.domsizes[rule]

    def save(self, filename :str) -> None:
        with open_to_write(filename) as fp:
            for rule in self.items:
                write_line(fp, (str(rule), self.domsizes[rule]))

    @staticmethod
    def load(filename :str) -> 'RuleSet':
        result = RuleSet()
        for rule_str, domsize in read_tsv_file(filename, types=(str, int)):
            rule = Rule.from_string(rule_str)
            result.add(rule, domsize)
        return result

# def load_ruleset(filename :str) -> Dict[str, Rule]:
#     result = {} # type: Dict[str, Rule]
#     for rule_str, freq, domsize in read_tsv_file(filename, (str, int, int)):
#         result[rule_str] = Rule.from_string(rule_str)
#         # TODO include domsize in the Rule object?
#     return result
# 
