import algorithms.mdl
from datastruct.lexicon import *
from datastruct.rules import *
from utils.files import *
from utils.printer import *
import settings

NUM_ITERATIONS = 99
GAMMA_THRESHOLD = 1e-30
RULES_FILE = 'rules.txt'
LEXICON_FILE = 'lexicon.txt'
TRAINING_LEXICON_FILE = 'lexicon.training'
ANALYSES_FILE = 'analyses.txt'

def expectation_maximization(lexicon, iter_count):
	# load rules and add the end-derivation-rule
	rules = RuleSet.load_from_file(RULES_FILE + '.' + str(iter_count-1))
	if not rules.has_key(u'#'):
		rules[u'#'] = RuleData(u'#', 1.0, 1.0, len(lexicon))
	print 'LogL =', lexicon.logl(rules)
	# build lexicon and reestimate parameters
	lexicon = algorithms.mdl.build_lexicon(rules, lexicon)
	algorithms.mdl.reestimate_rule_prod(rules, lexicon)
	algorithms.mdl.reestimate_rule_weights(rules, lexicon)
	return rules, lexicon

def save_analyses(lexicon, filename):
	with open_to_write(filename) as fp:
		for word in lexicon.values():
			write_line(fp, (word.word, '<- ' + ' <- '.join(word.analysis())))

def train_unsupervised():
	# optimize rules
	lexicon = Lexicon.init_from_file(settings.FILES['wordlist'])
	for i in range(1, NUM_ITERATIONS+1):
		print '\n===   Iteration %d   ===\n' % i
		rules, lexicon = expectation_maximization(lexicon, i)
		# save results
		rules.save_to_file(RULES_FILE + '.' + str(i))
		lexicon.save_to_file(LEXICON_FILE + '.' + str(i))
		save_analyses(lexicon, ANALYSES_FILE + '.' + str(i))

def train_supervised():
	rules, lexicon = algorithms.mdl.load_training_file(TRAINING_LEXICON_FILE)
	# compute rules domain size
	pp = progress_printer(len(rules))
	for rule_data in rules.values():
		rule = Rule.from_string(rule_data.rule)
		for word in lexicon.keys():
			if rule.lmatch(word):
				rule_data.domsize += 1
		pp.next()
	algorithms.mdl.reestimate_rule_prod(rules, lexicon)
	lexicon.save_to_file(LEXICON_FILE)
	rules.save_to_file(RULES_FILE)
	# optimize rules
	algorithms.optrules.optimize_rules_in_lexicon(\
		LEXICON_FILE,
		LEXICON_FILE + '.opt',
		RULES_FILE)
	rename_file(LEXICON_FILE + '.opt', LEXICON_FILE)
	lexicon = Lexicon.load_from_file(LEXICON_FILE)
	rules = RuleSet.load_from_file(RULES_FILE)
	if not rules.has_key(u'#'):
		rules[u'#'] = RuleData(u'#', 1.0, 1.0, len(lexicon))
	algorithms.mdl.reestimate_rule_weights(rules, lexicon)
	lexicon.save_to_file(LEXICON_FILE)
	rules.save_to_file(RULES_FILE)

def run():
	if settings.SUPERVISED:
		train_supervised()
	else:
		train_unsupervised()

def evaluate():
	pass

