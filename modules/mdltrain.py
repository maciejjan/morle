import algorithms.mdl
import algorithms.optrules
from datastruct.lexicon import *
from datastruct.rules import *
from utils.files import *
from utils.printer import *
import settings

GAMMA_THRESHOLD = 1e-30

def expectation_maximization(lexicon, rules, iter_count):
	# load rules and add the end-derivation-rule
#	rules = RuleSet.load_from_file(RULES_FILE + '.' + str(iter_count-1))
#	if not rules.has_key(u'#'):
#		rules[u'#'] = RuleData(u'#', 1.0, 1.0, len(lexicon))
	# build lexicon and reestimate parameters
	lexicon = algorithms.mdl.build_lexicon_new(rules, lexicon)
	algorithms.mdl.reestimate_rule_prod(rules, lexicon)
	if settings.USE_WORD_FREQ:
		algorithms.mdl.reestimate_rule_weights(rules, lexicon)
	logl = lexicon.logl(rules)
	print 'LogL =', logl
	algorithms.mdl.check_rules(rules, lexicon)
	return logl

def save_analyses(lexicon, filename):
	with open_to_write(filename) as fp:
		for word in lexicon.values():
			write_line(fp, (word.word, '<- ' + ' <- '.join(word.analysis_comp())))

def train_unsupervised():
#	algorithms.optrules.optimize_rules_in_graph(\
#		settings.FILES['training.wordlist'],\
#		settings.FILES['surface.graph'],\
#		settings.FILES['surface.graph'] + '.opt',\
#		settings.FILES['model.rules'] + '.0')
#	rename_file(settings.FILES['surface.graph'] + '.opt', settings.FILES['surface.graph'])
	lexicon = Lexicon.init_from_file(settings.FILES['training.wordlist'])
	rules = RuleSet()
#	rules[u'#'] = RuleData(u'#', 1.0, 1.0, len(lexicon))
#	for i in range(1, NUM_ITERATIONS+1):
	i = 0
	old_logl = lexicon.logl(rules)
	print 'null logl:', old_logl
	rules = RuleSet.load_from_file(settings.FILES['model.rules'] + '.0')
#	rules = RuleSet.load_from_file(settings.FILES['model.rules'])
	old_num_rules = len(rules)
	while True:
#	while i < 1:
		i += 1
		print '\n===   Iteration %d   ===\n' % i
		print 'number of rules:', old_num_rules
		logl = expectation_maximization(lexicon, rules, i)
		num_rules = len(rules)
		if num_rules == old_num_rules and logl - old_logl < 1.0:
#		if logl <= old_logl:
			break
		else:
			rules.save_to_file(settings.FILES['model.rules'])
			lexicon.save_to_file(settings.FILES['model.lexicon'])
			old_logl = logl
			old_num_rules = num_rules
	# save results
	save_analyses(lexicon, settings.FILES['analyses'])

def train_supervised():
	unigrams, rules, lexicon = algorithms.mdl.load_training_file(settings.FILES['training.lexicon'])
	unigrams.save_to_file(settings.FILES['model.ngrams'])
	# compute rules domain size
	pp = progress_printer(len(rules))
	for rule_data in rules.values():
		rule = Rule.from_string(rule_data.rule)
		for word in lexicon.keys():
			if rule.lmatch(word):
				rule_data.domsize += 1
		pp.next()
	algorithms.mdl.reestimate_rule_prod(rules, lexicon)
	lexicon.save_to_file(settings.FILES['model.lexicon'])
	rules.save_to_file(settings.FILES['model.rules'])
	# optimize rules
	algorithms.optrules.optimize_rules_in_lexicon(\
		settings.FILES['model.lexicon'],
		settings.FILES['model.lexicon'] + '.opt',
		settings.FILES['model.rules'])
	rename_file(settings.FILES['model.lexicon'] + '.opt',\
		settings.FILES['model.lexicon'])
	lexicon = Lexicon.load_from_file(settings.FILES['model.lexicon'])
	rules = RuleSet.load_from_file(settings.FILES['model.rules'])
	if not rules.has_key(u'#'):
		rules[u'#'] = RuleData(u'#', 1.0, 1.0, len(lexicon))
	algorithms.mdl.reestimate_rule_weights(rules, lexicon)
	lexicon.save_to_file(settings.FILES['model.lexicon'])
	rules.save_to_file(settings.FILES['model.rules'])

def run():
	if settings.SUPERVISED:
		train_supervised()
	else:
		train_unsupervised()

def evaluate():
	pass

