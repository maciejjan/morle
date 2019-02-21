from morle.datastruct.lexicon import Lexicon, LexiconEntry
from morle.datastruct.rules import Rule
import morle.shared as shared

import hfst
import unittest

# fake config file
CONFIG = '''
[General]
encoding = utf-8
date_format = %%d.%%m.%%Y %%H:%%M
supervised = no

[FST]
transducer_type = 1
'''

shared.config.read_string(CONFIG)


class RuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = []
        self.rules.append(Rule.from_string(':/a:ä/en:t___<VVINF>:<VVFIN>'))
        self.rules.append(Rule.from_string(':/:zu/:___<VVINF>:<VVINF>'))
        self.rules.append(Rule.from_string(':{CAP}/en:ung___<VVINF>:<NN>'))

        self.assertEqual(self.rules[0].subst, \
                         (((), ()), (('a',), ('ä',)), (('e', 'n'), ('t',))))
        self.assertEqual(self.rules[1].subst, \
                         (((), ()), ((), ('z', 'u')), ((), ())))
        self.assertEqual(self.rules[2].subst, \
                         (((), ('{CAP}',)), (('e', 'n'), ('u', 'n', 'g'))))

    def test_str(self) -> None:
        self.assertEqual(str(self.rules[0]), ':/a:ä/en:t___<VVINF>:<VVFIN>')
        self.assertEqual(str(self.rules[1]), ':/:zu/:___<VVINF>:<VVINF>')
        self.assertEqual(str(self.rules[2]), ':{CAP}/en:ung___<VVINF>:<NN>')

    def test_eq(self) -> None:
        self.assertEqual(self.rules[0],
                         Rule.from_string(':/a:ä/en:t___<VVINF>:<VVFIN>'))
        self.assertEqual(self.rules[1],
                         Rule.from_string(':/:zu/:___<VVINF>:<VVINF>'))
        self.assertEqual(self.rules[2],
                         Rule.from_string(':{CAP}/en:ung___<VVINF>:<NN>'))

    def test_fst(self) -> None:

        def test_fst_for_rule(self, rule, inputs, outputs) -> None:
            tr = rule.to_fst()
            tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
            for input_, output_ in zip(inputs, outputs):
                self.assertEqual(sorted(tr.lookup(input_)), sorted(output_))

        test_fst_for_rule(self, self.rules[0], ['fahren<VVINF>'],
                          [(('fährt<VVFIN>', 0.0),)])
        test_fst_for_rule(self, self.rules[0],
                          ['fallen<VVINF>', 'heraustragen<VVINF>'],
                          [(('fällt<VVFIN>', 0.0),),
                           (('herausträgt<VVFIN>', 0.0),
                            ('heräustragt<VVFIN>', 0.0)
                           )])
        # test the treatment of multi-character symbols
        test_fst_for_rule(self, self.rules[1], ['{CAP}aufmachen<VVINF>'],
                          [(('{CAP}zuaufmachen<VVINF>', 0.0),
                            ('{CAP}azuufmachen<VVINF>', 0.0),
                            ('{CAP}auzufmachen<VVINF>', 0.0),
                            ('{CAP}aufzumachen<VVINF>', 0.0),
                            ('{CAP}aufmzuachen<VVINF>', 0.0),
                            ('{CAP}aufmazuchen<VVINF>', 0.0),
                            ('{CAP}aufmaczuhen<VVINF>', 0.0),
                            ('{CAP}aufmachzuen<VVINF>', 0.0),
                            ('{CAP}aufmachezun<VVINF>', 0.0) \
                           )])

    def test_compute_domsize(self) -> None:
        lexicon = Lexicon()
        lexicon.add(LexiconEntry('anwinkeln<VVINF>'))
        lexicon.add(LexiconEntry('machen<VVINF>'))
        lexicon.add(LexiconEntry('Sachen<NN>'))
        lexicon.add(LexiconEntry('lachen<VVINF>'))
        lexicon.add(LexiconEntry('Dörfern<NN>'))
        lex_fst = lexicon.to_fst()
        self.assertEqual(self.rules[0].compute_domsize(lex_fst), 2)
        self.assertEqual(self.rules[1].compute_domsize(lex_fst), 18)
        self.assertEqual(self.rules[2].compute_domsize(lex_fst), 2)


class RuleSetTest(unittest.TestCase):
    pass

