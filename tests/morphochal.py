import sys
sys.path.insert(0, '../../src/')
from datastruct.lexicon import *
lexicon = Lexicon.load_from_file('lexicon.txt')
lexicon.analyses_morphochal('analyses_morphochal.txt')
#lexicon.save_splits('analyses_morphochal.txt')

