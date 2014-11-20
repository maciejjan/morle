import sys
sys.path.insert(0, '../../src/')
from datastruct.lexicon import *
from datastruct.rules import *
lexicon = Lexicon.load_from_file('lexicon.txt')
rules = RuleSet.load_from_file('rules.txt')
#word = u'abarbeiten'
#lexicon[word].root().show_tree()
#lexicon[word].root().annotate_word_structure()
#lexicon[word].root().show_split_tree()
from algorithms.mdl import rebuild_lexicon
#rebuild_lexicon(lexicon, rules)
lexicon.save_splits('analyses_morphochal.txt')
#rebuild_tree(lexicon, rules, lexicon[word].root().word)
#lexicon[word].root().show_tree()
#print lexicon[u'bauarbeiten'].structure
#print lexicon[u'bauarbeiten'].split()
