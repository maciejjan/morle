#import incubator.cooc

#import algorithms.vectorsim
#algorithms.vectorsim.calculate_vector_sim('par_rul.txt', 'par_sim.txt')

#from incubator.stemming import *
#from utils.files import *
#sort_file('inflection.txt', key=2)
#stem_lexemes('inflection.txt', 'segmentations.txt')

#import incubator.train
#incubator.train.run()

#import incubator.derivation
#incubator.derivation.run()

#import incubator.parsim
#incubator.parsim.run()
#incubator.parsim.export_to_db()

#import incubator.ruleprob
#incubator.ruleprob.run()

#import incubator.extract_paradigms
#incubator.extract_paradigms.run()
#incubator.extract_paradigms.evaluate()

#import incubator.mdl
#incubator.mdl.run()
#incubator.mdl.em()
#d = incubator.mdl.load_graph('graph_mdl_fil.txt')

#import incubator.ngrams as ngr
#unigram = ngr.train('input.txt', 1)

#import incubator.mdl2
#incubator.mdl2.run()
#incubator.mdl2.compute_rule_costs('rules.txt.451', 'lexicon.txt.451', 'rc.txt')

#import incubator.ruleprob
#incubator.ruleprob.run()

import incubator.genwords
#incubator.genwords.run()

#import incubator.optrules
#incubator.optrules.run()
#rc, applying, matching = incubator.optrules.test(u':/b:us/:')
#opt = incubator.optrules.test(u':/b:us/:')

def print_opt(opt):
	for rule, freq, domsize, wordpairs in opt:
		print ''
		print '\t'.join((rule, str(freq), str(domsize)))
		for w1, w2 in wordpairs:
			print '\t'.join((w1, w2))

