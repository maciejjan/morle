from datastruct.lexicon import LexiconEntry
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

[Features]
word_vec_dim = 100
'''

shared.config.read_string(CONFIG)

class LexiconEntryTest(unittest.TestCase):
    # literal, word, tag, disamb, symstr
    test_examples = [\
        ('EXAMPLE<NN><SG>', 
         ['{ALLCAPS}', 'e', 'x', 'a', 'm', 'p', 'l', 'e'], 
         ['<NN>', '<SG>'], None, '{ALLCAPS}example<NN><SG>')
    ]

    def test_init(self) -> None:

        for literal, word, tag, disamb, symstr in \
                LexiconEntryTest.test_examples:
            lexentry = LexiconEntry(literal)
            self.assertEqual(lexentry.literal, literal)
            self.assertEqual(lexentry.word, word)
            self.assertEqual(lexentry.tag, tag)
            self.assertEqual(lexentry.disamb, disamb)
            self.assertEqual(lexentry.symstr, symstr)

    def test_copy(self) -> None:
        for literal, word, tag, disamb, symstr in \
                LexiconEntryTest.test_examples:
            lexentry = LexiconEntry(literal)
            self.assertEqual(lexentry, lexentry.copy())

