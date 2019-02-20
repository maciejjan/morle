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

[preprocess]
max_num_rules = 5000
min_rule_freq = 3
max_edges_per_wordpair = 3
min_edges_per_wordpair = 1
max_affix_length = 5
max_infix_length = 3
max_infix_slots = 1
'''

shared.config.read_string(CONFIG)
