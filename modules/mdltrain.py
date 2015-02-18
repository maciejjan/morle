import algorithms.mdl
import algorithms.optrules
from datastruct.lexicon import *
from datastruct.rules import *
from utils.files import *
from utils.printer import *
import settings

def save_analyses(lexicon, filename):
	with open_to_write(filename) as fp:
		for word in sorted(lexicon.values(), key=lambda x:x.word):
			write_line(fp, (word.word, '<- ' + ' <- '.join(word.analysis())))

def expectation_maximization(lexicon):
	algorithms.mdl.optimize_lexicon(lexicon)
	algorithms.mdl.optimize_rule_params(lexicon)
	logl = lexicon.logl()
	print('LogL =', round(logl))
	algorithms.mdl.check_rules(lexicon)
	return logl

def train_unsupervised():
	ruleset = RuleSet.load_from_file(settings.FILES['model.rules'] + '.0')
	lexicon = Lexicon.init_from_file(settings.FILES['training.wordlist'],\
		ruleset=ruleset)
	i = 0
	old_logl = lexicon.logl()
	print('null logl:', round(old_logl))

	old_num_rules = len(lexicon.ruleset)
	while True:
		i += 1
		print('\n===   Iteration %d   ===\n' % i)
		print('number of rules:', old_num_rules)
		logl = expectation_maximization(lexicon)
		num_rules = len(lexicon.ruleset)
		if num_rules == old_num_rules and logl - old_logl < 1.0:
			break
		else:
			lexicon.ruleset.save_to_file(settings.FILES['model.rules'])
			lexicon.save_to_file(settings.FILES['model.lexicon'])
			old_logl = logl
			old_num_rules = num_rules
	# save results
	save_analyses(lexicon, settings.FILES['analyses'])

def train_supervised():
	lexicon = algorithms.mdl.load_training_file(settings.FILES['training.lexicon'])
#	unigrams.save_to_file(settings.FILES['model.ngrams'])
	# compute rules domain size -- TODO
	trh = TrigramHash()
	for word in lexicon.keys():
		trh.add(word)
	print('Calculating rule domain sizes...')
	pp = progress_printer(len(lexicon.ruleset))
	for r in lexicon.ruleset.values():
		r.domsize = algorithms.optrules.rule_domsize(r.rule, trh)
		next(pp)
	algorithms.mdl.optimize_rule_params(lexicon)
	lexicon.save_model(settings.FILES['model.rules'], settings.FILES['model.lexicon'])
	# optimize rules
	algorithms.optrules.optimize_rules_in_lexicon(\
		settings.FILES['model.lexicon'],
		settings.FILES['model.lexicon'] + '.opt',
		settings.FILES['model.rules'])
	rename_file(settings.FILES['model.lexicon'] + '.opt',\
		settings.FILES['model.lexicon'])
	ruleset = RuleSet.load_from_file(settings.FILES['model.rules'])
	lexicon = Lexicon.load_from_file(settings.FILES['model.lexicon'], ruleset)
	algorithms.mdl.optimize_rule_params(lexicon)
	lexicon.save_model(settings.FILES['model.rules'], settings.FILES['model.lexicon'])

def run():
	if settings.SUPERVISED:
		train_supervised()
	else:
		train_unsupervised()

def evaluate():
	pass

