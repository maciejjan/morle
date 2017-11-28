from algorithms.negex import NegativeExampleSampler
from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.rules import Rule, RuleSet
from datastruct.graph import GraphEdge, EdgeSet
import shared

import unittest


# fake config file
CONFIG = '''
[General]
encoding = utf-8
date_format = %%d.%%m.%%Y %%H:%%M
supervised = no

[Models]
root_feature_model = none
edge_feature_model = none

[NegativeExampleSampler]
num_processes = 1
'''

shared.config.read_string(CONFIG)


class NegativeExampleSamplerTest(unittest.TestCase):

    def test_complete_sample(self) -> None:
        'Test a sample consisting of all possible negative edges.'
        words = ['machen', 'macht', 'mache', 'Sachen', 'Sache',
                 'anwinkeln', 'anzuwinkeln']
        rules = [\
            ':/en:t___:',
            ':/n:___:',
            ':/a:ä/:er___:',
            ':/:zu/:___:'
        ]
        positive_edges = [\
            ('machen', 'macht', ':/en:t___:'),
            ('machen', 'mache', ':/n:___:'),
            ('Sachen', 'Sache', ':/n:___:'),
            ('anwinkeln', 'anzuwinkeln', ':/:zu/:___:'),
        ]
        expected_negative_edges = [\
            ('Sachen', '{CAP}sacht', ':/en:t___:'),
            ('anwinkeln', 'anwinkel', ':/n:___:'),
            ('anzuwinkeln', 'anzuwinkel', ':/n:___:'),
            ('machen', 'mächener', ':/a:ä/:er___:'),
            ('macht', 'mächter', ':/a:ä/:er___:'),
            ('mache', 'mächeer', ':/a:ä/:er___:'),
            ('Sachen', '{CAP}sächener', ':/a:ä/:er___:'),
            ('Sache', '{CAP}sächeer', ':/a:ä/:er___:'),
            ('machen', 'mzuachen', ':/:zu/:___:'),
            ('machen', 'mazuchen', ':/:zu/:___:'),
            ('machen', 'maczuhen', ':/:zu/:___:'),
            ('machen', 'machzuen', ':/:zu/:___:'),
            ('machen', 'machezun', ':/:zu/:___:'),
            ('mache', 'mzuache', ':/:zu/:___:'),
            ('mache', 'mazuche', ':/:zu/:___:'),
            ('mache', 'maczuhe', ':/:zu/:___:'),
            ('mache', 'machzue', ':/:zu/:___:'),
            ('macht', 'mzuacht', ':/:zu/:___:'),
            ('macht', 'mazucht', ':/:zu/:___:'),
            ('macht', 'maczuht', ':/:zu/:___:'),
            ('macht', 'machzut', ':/:zu/:___:'),
            ('Sachen', '{CAP}zusachen', ':/:zu/:___:'),
            ('Sachen', '{CAP}szuachen', ':/:zu/:___:'),
            ('Sachen', '{CAP}sazuchen', ':/:zu/:___:'),
            ('Sachen', '{CAP}saczuhen', ':/:zu/:___:'),
            ('Sachen', '{CAP}sachzuen', ':/:zu/:___:'),
            ('Sachen', '{CAP}sachezun', ':/:zu/:___:'),
            ('Sache', '{CAP}zusache', ':/:zu/:___:'),
            ('Sache', '{CAP}szuache', ':/:zu/:___:'),
            ('Sache', '{CAP}sazuche', ':/:zu/:___:'),
            ('Sache', '{CAP}saczuhe', ':/:zu/:___:'),
            ('Sache', '{CAP}sachzue', ':/:zu/:___:'),
            ('anwinkeln', 'azunwinkeln', ':/:zu/:___:'),
            ('anwinkeln', 'anwzuinkeln', ':/:zu/:___:'),
            ('anwinkeln', 'anwizunkeln', ':/:zu/:___:'),
            ('anwinkeln', 'anwinzukeln', ':/:zu/:___:'),
            ('anwinkeln', 'anwinkzueln', ':/:zu/:___:'),
            ('anwinkeln', 'anwinkezuln', ':/:zu/:___:'),
            ('anwinkeln', 'anwinkelzun', ':/:zu/:___:'),
            ('anzuwinkeln', 'azunzuwinkeln', ':/:zu/:___:'),
            ('anzuwinkeln', 'anzuzuwinkeln', ':/:zu/:___:'),
            ('anzuwinkeln', 'anzzuuwinkeln', ':/:zu/:___:'),
            ('anzuwinkeln', 'anzuwzuinkeln', ':/:zu/:___:'),
            ('anzuwinkeln', 'anzuwizunkeln', ':/:zu/:___:'),
            ('anzuwinkeln', 'anzuwinzukeln', ':/:zu/:___:'),
            ('anzuwinkeln', 'anzuwinkzueln', ':/:zu/:___:'),
            ('anzuwinkeln', 'anzuwinkezuln', ':/:zu/:___:'),
            ('anzuwinkeln', 'anzuwinkelzun', ':/:zu/:___:')
        ]
        expected_weights = {\
            ':/en:t___:' : 1.0,
            ':/n:___:' : 1.0,
            ':/a:ä/:er___:' : 1.0,
            ':/:zu/:___:' : 41/40       # the word "anzuzuwinkeln" can be
                                        # derived in two different ways, so
                                        # it is counted double in domsize
                                        # computation, but sampled only once;
                                        # such cases are very rare, so they
                                        # shouldn't influence the weights much
        }

        lexicon = Lexicon(LexiconEntry(word) for word in words)
        lex_fst = lexicon.to_fst()
        rule_set = RuleSet()
        for rule_str in rules:
            rule = Rule.from_string(rule_str)
            rule_set.add(rule, rule.compute_domsize(lex_fst))
        edge_iter = (GraphEdge(lexicon[source], lexicon[target],
                               rule_set[rule]) \
                     for (source, target, rule) in positive_edges)
        edge_set = EdgeSet(lexicon, edge_iter)
                           
        negex_sampler = NegativeExampleSampler(rule_set)
        sample_size = len(expected_negative_edges)
        sample = negex_sampler.sample(lexicon, sample_size)
        sample_weights = negex_sampler.compute_sample_weights(sample, edge_set)

        self.assertEqual(rule_set.get_domsize(rule_set[0]), 2)
        self.assertEqual(rule_set.get_domsize(rule_set[1]), 4)
        self.assertEqual(rule_set.get_domsize(rule_set[2]), 5)
        self.assertEqual(rule_set.get_domsize(rule_set[3]), 42)
        self.longMessage=False
        for edge in edge_set:
            self.assertNotIn(edge, sample,
                             msg='positive edge: {} in sample'.format(edge))
        for source, target, rule in expected_negative_edges:
            edge = GraphEdge(lexicon[source], LexiconEntry(target),
                             rule_set[rule])
            self.assertIn(edge, sample, msg='{} not in sample'.format(edge))
        self.longMessage=True
        for i, edge in enumerate(sample):
            self.assertAlmostEqual(sample_weights[i],
                                   expected_weights[str(edge.rule)], 
                                   msg='for edge {}'.format(edge))
                                
    def test_random_sample(self) -> None:
        'Test a random sample in case there are many possible negative edges.'
        pass
        # TEST DATA:
        # Lexicon:
        # RuleSet:
        # positive edges:
        # test for a valid negative edge:
        # - source in lexicon
        # - target not in lexicon
        # - rule in RuleSet
        # - "target" in transducer source .o. rule
        # - weights: [TODO values]

