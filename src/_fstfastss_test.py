from algorithms.fst import *
from algorithms.fstfastss import *
import configparser

tr = load_transducer('/disk/data/morle/pol-unsup/test/lexicon.fsm')
shared.config = configparser.ConfigParser()
shared.config.read('/disk/data/morle/pol-unsup/test/config.ini')
shared.options['working_dir'] = '/disk/data/morle/pol-unsup/test/'

# tr_l, tr_r = similar_words(tr)
sw_list = []
sw = similar_words(tr)
for i in range(100000):
    sw_list.append(next(sw))

