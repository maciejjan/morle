from algorithms.fst import *
from algorithms.fstfastss import *
import configparser

tr = load_transducer('/disk/data/morle/pol-unsup/1/lexicon.fsm')
shared.config = configparser.ConfigParser()
shared.config.read('/disk/data/morle/pol-unsup/1/config.ini')

sw = similar_words(tr)

