#import algorithms.align
import algorithms.fst
from utils.files import *
import libhfst
import re
#from scipy.stats import norm
from algorithms.ngrams import *
from operator import itemgetter
#import math

class Rule:
    def __init__(self, subst, tag_subst=None, string=None):
        self.subst = subst
        self.tag_subst = tag_subst
        self.transducer = None
        if string is None:
            self.string = self.to_string()
        else:
            self.string = string
        self.left_pattern = None
        
    def __le__(self, other):
        pattern = 'X'.join(''.join(s[0]) for s in self.subst) +\
            (''.join(self.tag_subst[0]) if self.tag_subst else '')
        return other.lmatch_string(pattern)
    
    def __eq__(self, other):
        return self.subst == other.subst and self.tag_subst == other.tag_subst
    
    def __str__(self):
        return self.string
    
    def __hash__(self):
        return self.string.__hash__()
    
    def copy(self):
        return Rule(self.subst, self.tag_subst)
    
    def apply(self, node):
        if self.transducer is None:
            self.build_transducer()
        t = algorithms.fst.seq_to_transducer(node.seq())
        t.compose(self.transducer)
        t.determinize()
        t.minimize()
        return list(map(itemgetter(0), sum(t.extract_paths().values(), [])))
    
    # TODO more meaningful name
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
    
    def compute_domsize(self, lexicon):
        if lexicon.transducer is None:
            lexicon.build_transducer()
        self.build_transducer(alphabet=lexicon.alphabet)
        t = libhfst.HfstTransducer(lexicon.transducer)
        t.compose(self.transducer)
        t.determinize()
        t.minimize()
        if t.is_cyclic():
            print('Warning: cyclic transducer for %s' % self.__str__())
#        return t.number_of_states()
        return algorithms.fst.number_of_paths(t)
#        return len(list(map(itemgetter(0), sum(t.extract_paths(max_cycles=1).values(), []))))

#    def compute_domsize_2(self, lexicon):
#        if lexicon.transducer is None:
#            lexicon.build_transducer()
#        self.build_transducer(alphabet=lexicon.alphabet)
#        t = libhfst.HfstTransducer(lexicon.transducer)
#        t.compose(self.transducer)
#        t.determinize()
#        t.minimize()
##        t.lookup_optimize()
##        for node in lexicon.iter_nodes():
##            print(t.lookup(node.key))
##            break
#        return sum(len(t.lookup(node.key)) for node in lexicon.iter_nodes())

    def get_trigrams(self):
        trigrams = []
        for tr in generate_n_grams(('^',)+self.subst[0][0]+(libhfst.IDENTITY,), 3):
            if len(tr) == 3:
                trigrams.append(tr)
        for alt in self.subst[1:-1]:
            for tr in generate_n_grams((libhfst.IDENTITY,)+alt[0]+(libhfst.IDENTITY,), 3):
                if len(tr) == 3:
                    trigrams.append(tr)
        for tr in generate_n_grams((libhfst.IDENTITY,)+self.subst[-1][0]+('$',), 3):
            if len(tr) == 3:
                trigrams.append(tr)
        return trigrams

    def seq(self):
        x_seq, y_seq = [], []
        x_tseq, y_tseq = [], []
        for x, y in self.subst:
            x_seq.extend(x + (libhfst.EPSILON,)*(len(y)-len(x)))
            x_seq.append(libhfst.IDENTITY)
            y_seq.extend((libhfst.EPSILON,)*(len(x)-len(y)) + y)
            y_seq.append(libhfst.IDENTITY)
        x_seq.pop()        # remove the last identity symbol
        y_seq.pop()
        if self.tag_subst:
            xt, yt = self.tag_subst
            x_seq.extend(xt + (libhfst.EPSILON,)*(len(yt)-len(xt)))
            y_seq.extend((libhfst.EPSILON,)*(len(xt)-len(yt)) + yt)
        return tuple(zip(x_seq, y_seq))
    
    def input_seq(self):
        seq = []
        for x, y in self.subst[:-1]:
            seq.extend(x)
            if not seq or seq[-1] != libhfst.IDENTITY:
                seq.append(libhfst.IDENTITY)
        seq.extend(self.subst[-1][0])
        if self.tag_subst:
            seq.extend(self.tag_subst[0])
        return tuple(seq)
    
    def ngrams(self):
        pass
    
    @staticmethod
    def from_seq(seq, tag_subst):
        subst = []
        x_seq, y_seq = (), ()
        for x, y in seq:
            if x == y == libhfst.IDENTITY:
                subst.append((x_seq, y_seq))
                x_seq, y_seq = (), ()
            else:
                x_seq += (x,) if x != libhfst.EPSILON else ()
                y_seq += (y,) if y != libhfst.EPSILON else ()
        subst.append((x_seq, y_seq))
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
    
    def build_transducer(self, alphabet=None):
        self.transducer =\
            algorithms.fst.seq_to_transducer(self.seq(), alphabet=alphabet)
    
    def lmatch(self, lexicon):
        if lexicon.transducer is None:
            lexicon.build_transducer()
        if self.transducer is None:
            self.build_transducer()
        t = libhfst.HfstTransducer(lexicon.transducer)
        t.compose(self.transducer)
        t.input_project()
        t.determinize()
        t.minimize()
        return tuple(t.extract_paths().keys())
    
    def rmatch(self, lexicon):
        if lexicon.transducer is None:
            lexicon.build_transducer()
        if self.transducer is None:
            self.build_transducer()
        t = libhfst.HfstTransducer(self.transducer)
        t.compose(lexicon.transducer)
        t.output_project()
        t.determinize()
        t.minimize()
        return tuple(t.extract_paths().keys())
    
    def lmatch_node(self, node):
        return self.lmatch_string(node.key())

    def lmatch_string(self, string):
        if not self.left_pattern:
            pattern = '.+'.join(''.join(s[0]) for s in self.subst)
            if self.tag_subst and self.tag_subst[0]:
                pattern += ''.join(self.tag_subst[0])
#            print(pattern)
            self.left_pattern = re.compile(pattern)
        return True if self.left_pattern.match(string) else False

#        if self.transducer is None:
#            self.build_transducer()
#        t = algorithms.fst.seq_to_transducer(node.seq())
#        t.compose(self.transducer)
#        return True if t.extract_paths() else False

#def tokenize_word(string):
#    '''Separate a string into a word and a POS-tag,
#       both expressed as sequences of symbols.'''
#    m = re.match(settings.WORD_PATTERN_CMP, string)
#    if m is None:
#        raise Exception('Error while tokenizing word: %s' % string)
#    return tuple(re.findall(settings.SYMBOL_PATTERN_CMP, m.group('word'))),\
#           tuple(re.findall(settings.TAG_PATTERN_CMP, m.group('tag')))

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
#        mt = re.match(settings.

#        p = string.find('___')
#        tag = None
#        if p > -1:
#            tag_sp = string[p+3:].split(':')
#            tag = (tag_sp[0], tag_sp[1])
#            string = string[:p]
#        split = string.split('/')
#        prefix_sp = split[0].split(':')
#        prefix = (prefix_sp[0], prefix_sp[1])
#        suffix_sp = split[-1].split(':')
#        suffix = (suffix_sp[0], suffix_sp[1])
#        alternations = []
#        for alt_str in split[1:-1]:
#            alt_sp = alt_str.split(':')
#            alternations.append((alt_sp[0], alt_sp[1]))
#        return Rule(prefix, alternations, suffix, tag)


class RuleSetItem:
#    def __init__(self, rule, prod, freqcl, domsize, vec_mean, vec_sd):
    def __init__(self, rule, prod, freqcl, domsize):
        self.rule = rule
        self.rule_obj = Rule.from_string(rule)
        self.prod = prod
        self.freqcl = freqcl
        self.domsize = domsize
#        self.vec_mean = vec_mean
#        self.vec_sd = vec_sd
    
    def __lt__(self, other):
        return self.rule < other.rule
    
    def freqprob(self, f):
        p = 2 * norm.pdf(f-self.freqcl) * norm.cdf(f-self.freqcl)
        return p if p > 0 else 1e-100
#        return norm.pdf(f, self.freqcl, 1)
    
class RuleSet:
    def __init__(self):
        self.rsp = RuleSetPrior()
        self.rules = {}
    
    def __len__(self):
        return len(self.rules)
    
    def __delitem__(self, key):
        del self.rules[key]
    
    def keys(self):
        return self.rules.keys()

    def values(self):
        return self.rules.values()
    
    def items(self):
        return self.rules.items()
    
    def __contains__(self, key):
        return self.rules.__contains__(key)
    
    def __getitem__(self, key):
        return self.rules[key]

    def __setitem__(self, key, val):
        self.rules[key] = val
    
    def rule_cost(self, rule, freq):
        result = math.log(self.rsp.rule_prob(rule))
        r = self.rules[rule]
        result += freq * math.log(r.prod)
        result += (r.domsize-freq) * math.log(1-r.prod)
        return result

    def filter(self, condition):
        keys_to_delete = []
        for key, val in self.rules.items():
            if not condition(val):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.rules[key]
    
    def save_to_file(self, filename):
        with open_to_write(filename) as fp:
            for r_str, r in self.rules.items():
                write_line(fp, (r_str, r.prod, r.freqcl, r.domsize))
    
    @staticmethod
    def load_from_file(filename):
        rs = RuleSet()
        for rule, prod, freqcl, domsize in read_tsv_file(filename,\
                types=(str, float, int, int)):
            rs[rule] = RuleData(rule, prod, freqcl, domsize)
        return rs
    

def align_parts(left, right):
    lcs = algorithms.align.lcs(left, right)
    pattern = re.compile('(.*)' + '(.*?)'.join([\
        letter for letter in lcs]) + '(.*)')
    m1 = pattern.search(left)
    m2 = pattern.search(right)
    alignment = []
    for i, (x, y) in enumerate(zip(m1.groups(), m2.groups())):
        if x or y:
            alignment.append((x, y))
        if i < len(lcs):
            alignment.append((lcs[i], lcs[i]))
    return alignment


class RuleSetPrior:
    def __init__(self):
        self.ngr_add = None
        self.ngr_remove = None
        self.unigrams = NGramModel(1)
        filename = settings.FILES['training.lexicon'] if settings.SUPERVISED \
            else settings.FILES['training.wordlist']
        if settings.USE_WORD_FREQ:
            self.unigrams.train_from_file(filename)
        else:
            self.unigrams.train([(word, 1) for (word, ) in read_tsv_file(
                filename, (str, ))])

    def train(self, rules_c):
        self.ngr_add = NGramModel(1)
        self.ngr_remove = NGramModel(1)
        ngram_training_pairs_add = []
        ngram_training_pairs_remove = []
        for rule_str, count in rules_c.items():
            if rule_str.count('*') > 0:
                continue
            rule = Rule.from_string(rule_str)
#            ngram_training_pairs.extend([\
#                (rule.prefix[0], count), (rule.suffix[0], count),\
#                (rule.suffix[0], count), (rule.suffix[1], count)
#            ])
            ngram_training_pairs_remove.extend([\
                (rule.prefix[0], count), (rule.suffix[0], count)])
            ngram_training_pairs_add.extend([\
                (rule.prefix[1], count), (rule.suffix[1], count)])
            for x, y in rule.alternations:
                ngram_training_pairs_remove.append((x, count))
                ngram_training_pairs_add.append((y, count))
            # an 'empty alternation' finishes the sequence of alternations
            ngram_training_pairs_remove.append((u'', count))
            ngram_training_pairs_add.append((u'', count))
        self.ngr_add.train(ngram_training_pairs_add)
        self.ngr_remove.train(ngram_training_pairs_remove)
    
    def rule_prob(self, rule_str):
        rule = Rule.from_string(rule_str)
        al = align_parts(rule.prefix[0], rule.prefix[1])
        if rule.alternations:
            al.append(('.', '.'))
            for x, y in rule.alternations:
                al.extend(align_parts(x, y) + [('.', '.')])
        al.extend(align_parts(rule.suffix[0], rule.suffix[1]))
        p = 1
#        print(al)
        for x, y in al:
            if x == '.' and y == '.':    # PASS
                p *= 0.5
            elif x == y:    # CHECK
                p *= 0.4**len(x) * self.unigrams.ngram_prob(x)
            elif not x and y:    # INSERT
                p *= 0.09999**len(y) * self.unigrams.ngram_prob(y)
            elif x and not y:
                p *= 0.00001**len(x) * self.unigrams.ngram_prob(x)
            else:
                p *= 0.00001**len(x) * self.unigrams.ngram_prob(x)
                p *= 0.09999**len(y) * self.unigrams.ngram_prob(y)
        # TODO parameters probability
#        p *= 
        return p
    
    def rule_prob_old(self, rule_str):
        if rule_str == u':/:':
            return 0.0
        rule = Rule.from_string(rule_str)
        prob = self.ngr_remove.word_prob(rule.prefix[0]) *\
            self.ngr_add.word_prob(rule.prefix[1]) *\
            self.ngr_remove.word_prob(rule.suffix[0]) *\
            self.ngr_add.word_prob(rule.suffix[1]) *\
            self.ngr_remove.word_prob(u'') * self.ngr_add.word_prob(u'')
        for x, y in rule.alternations:
            prob *= self.ngr_remove.word_prob(x) * self.ngr_add.word_prob(y)
        prob /= 1.0 - self.ngr_add.word_prob(u'') ** 3 * self.ngr_remove.word_prob(u'') ** 3        # empty rule not included
        return prob
    
    def param_cost(self, prod, weight):
        result = 0.0
#        print prod, math.log(prod), round(math.log(prod))
        result -= math.log(0.5) * round(math.log(prod))
#        print result
        result += math.log(2 * norm.pdf(weight, 2, 1))
#        print result
        return result

