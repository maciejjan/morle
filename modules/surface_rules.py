import algorithms.align
import algorithms.fastss
import algorithms.cooccurrences
from datastruct.counter import *
from datastruct.rules import *
import utils.db
from utils.files import *
from utils.printer import *
import settings
import re

### LOCAL FILTERS ###

# form: filter(rule, (wordpair))

def _lfil_affix_length(rule, wordpair):
	affixes = rule.get_affixes()
	if not affixes:
		return True
	if len(affixes) == 1 and not rule.alternations: # exception for single-affix rules
		return True
	if max([len(a) for a in affixes]) <= settings.MAX_AFFIX_LENGTH:
		return True
	return False

LOCAL_FILTERS = [_lfil_affix_length]

def apply_local_filters(rule, wordpair):
	for f in LOCAL_FILTERS:
		if not f(rule, wordpair):
			return False
	return True

### GLOBAL FILTERS ###

# form: filter(rule, [list, of, wordpairs...]) 
def _gfil_rule_freq(rule, wordpairs):
	return len(wordpairs) >= settings.MIN_RULE_FREQ

GLOBAL_FILTERS = [_gfil_rule_freq]

def apply_global_filters(rule, wordpairs):
	for f in GLOBAL_FILTERS:
		if not f(rule, wordpairs):
			return False
	return True

### TRAINING DATA ###

def load_training_infl_rules(filename):
	i_rules_c = {}
	print 'Loading inflectional rules...'
	for rule, ifreq, freq, weight in read_tsv_file(filename):
		i_rules_c[rule] = weight
	return i_rules_c

### RULE EXTRACTING FUNCTIONS ###

def extract_rules_from_words(words, substring, outfp, i_rules_c):
	pattern = re.compile('(.*)' + '(.*?)'.join([letter for letter in substring]) + '(.*)')
	for w1, w1_freq in words:
		for w2, w2_freq in words:
			if w1 < w2:
				rule = algorithms.align.extract_rule(w1, w2, pattern)
				if rule is not None and (i_rules_c is None or i_rules_c.has_key(rule.to_string())):
					if apply_local_filters(rule, ((w1, w1_freq), (w2, w2_freq))):
						write_line(outfp, (w1, w2, rule.to_string()))
						write_line(outfp, (w2, w1, rule.reverse().to_string()))

def extract_rules_from_substrings(input_file, output_file, i_rules_c=None):
	cur_substr, words = '', []
	pp = progress_printer(get_file_size(input_file))
	print 'Extracting rules from substrings...'
	with open_to_write(output_file) as outfp:
		for s_len, substr, word, freq in read_tsv_file(input_file):	# TODO _by_key
			if substr != cur_substr:
				if len(words) > 1:
					extract_rules_from_words(words, cur_substr, outfp, i_rules_c)
				cur_substr = substr
				words = [(word, int(freq))]
			else:
				words.append((word, int(freq)))
			pp.next()
		if len(words) > 1:
			extract_rules_from_words(words, cur_substr, outfp, i_rules_c)

def filter_and_count_rules(input_file):
	rules_c = Counter()
	output_file = input_file + '.filtered'
	lines_written = 0
	print 'Filtering and counting rules...'
	with open_to_write(output_file) as outfp:
		pp = progress_printer(get_file_size(input_file))
		for rule_str, wordpairs in read_tsv_file_by_key(input_file, 3):
			rule = Rule.from_string(rule_str)
			if apply_global_filters(rule, wordpairs):
				for (w1, w2) in wordpairs:
					write_line(outfp, (w1, w2, rule_str))
					lines_written += 1
				rules_c.inc(rule_str, len(wordpairs))
			for i in range(0, len(wordpairs)):
				pp.next()
	remove_file(input_file)
	rename_file(output_file, input_file)
	set_file_size(input_file, lines_written)
	return rules_c

### WEIGHTING RULES ACCORDING TO PROBABILITY ###

def calculate_rule_prob(input_file, rules_c):
	patterns, counts = {}, Counter()
	for r in rules_c.keys():
		rule = Rule.from_string(r)
		patterns[r] = re.compile(\
			('^' + rule.prefix[0] + '.*' + \
			'.*'.join([a[0] for a in rule.alternations]) + '.*' + rule.suffix[0] + '$')\
			.replace('.*.*', '.*'))
	print 'Calculating rule probability...'
	pp = progress_printer(get_file_size(input_file))
	for word, freq in read_tsv_file(input_file):
		for r in rules_c.keys():
			if re.match(patterns[r], word):
				counts.inc(r)
		pp.next()
	ruleprob = {}
	for r in rules_c.keys():
		ruleprob[r] = float(rules_c[r]) / counts[r]
	return ruleprob

def save_rules(rules_c, ruleprob, filename):
	print 'Saving rules...'
	with open_to_write(filename) as fp:
		for r in rules_c.keys():
			write_line(fp, (r, rules_c[r], ruleprob[r], 0.0))

### MAIN FUNCTIONS ###

def run():
	algorithms.fastss.create_substrings_file(\
		settings.FILES['training.wordlist'], settings.FILES['surface.substrings'])
	if file_exists(settings.FILES['trained.rules']):
		extract_rules_from_substrings(settings.FILES['surface.substrings'],\
			settings.FILES['surface.graph'],\
			load_training_infl_rules(settings.FILES['trained.rules']))
	else:
		extract_rules_from_substrings(settings.FILES['surface.substrings'],\
			settings.FILES['surface.graph'])
	sort_file(settings.FILES['surface.graph'], key=(1,2), unique=True)
	sort_file(settings.FILES['surface.graph'], key=3)
	update_file_size(settings.FILES['surface.graph'])
	rules_c = filter_and_count_rules(settings.FILES['surface.graph'])
	rules_c.save_to_file(settings.FILES['surface.rules'])
#	ruleprob = calculate_rule_prob(settings.FILES['wordlist'], rules_c)
#	save_rules(rules_c, ruleprob, settings.FILES['surface.rules'])
#	sort_file(settings.FILES['surface.graph'], key=1)
#	rules_c = Counter.load_from_file(settings.FILES['surface.rules'])
#	if not file_exists(settings.FILES['trained.rules.cooc']):
#		algorithms.cooccurrences.calculate_rules_cooc(\
#			settings.FILES['surface.graph'],\
#			settings.FILES['surface.rules.cooc'], rules_c)

def evaluate():
	print '\nSurface rules: nothing to evaluate.\n'

def import_from_db():
	utils.db.connect()
	print 'Importing wordlist...'
	utils.db.pull_table(settings.WORDS_TABLE, ('word', 'freq'),\
		settings.FILES['training.wordlist'])
	print 'Importing surface rules...'
	utils.db.pull_table(settings.S_RUL_TABLE, ('rule', 'freq', 'prob'),\
		settings.FILES['surface.rules'])
	# pull graph
	print 'Importing graph...'
	with open_to_write(settings.FILES['surface.graph']) as fp:
		for word_1, word_2, rule in utils.db.query_fetch_results('''
			SELECT w1.word, w2.word, r.rule FROM graph g 
				JOIN words w1 ON g.w1_id = w1.w_id
				JOIN words w2 ON g.w2_id = w2.w_id
				JOIN s_rul r ON g.r_id = r.r_id
			;'''):
			write_line(fp, (word_1, word_2, rule))
	# pull surface rules co-occurrences
	print 'Importing surface rules co-occurrences...'
	with open_to_write(settings.FILES['surface.rules.cooc']) as fp:
		for rule_1, rule_2, freq, sig in utils.db.query_fetch_results('''
			SELECT r1.rule, r2.rule, c.freq, c.sig FROM s_rul_co c
				JOIN s_rul r1 ON c.r1_id = r1.r_id
				JOIN s_rul r2 ON c.r2_id = r2.r_id
			;'''):
			write_line(fp, (rule_1, rule_2, freq, sig))
	utils.db.close_connection()

def export_to_db():
	# words <- insert ID
	print 'Converting wordlist...'
	word_ids = utils.db.insert_id(settings.FILES['training.wordlist'],\
		settings.FILES['wordlist.db'])
	# surface rules <- insert ID
	print 'Converting surface rules...'
	s_rule_ids = utils.db.insert_id(settings.FILES['surface.rules'],\
		settings.FILES['surface.rules.db'])
	# graph <- replace words and surface rules with their ID
	print 'Converting graph...'
	utils.db.replace_values_with_ids(settings.FILES['surface.graph'],\
		settings.FILES['surface.graph.db'],\
		(word_ids, word_ids, s_rule_ids))
	# surface_rules_cooc <- replace rules with ID
	print 'Converting surface rules co-occurrences...'
	utils.db.replace_values_with_ids(settings.FILES['surface.rules.cooc'],\
		settings.FILES['surface.rules.cooc.db'],\
		(s_rule_ids, s_rule_ids, None, None))
	# load tables into DB
	utils.db.connect()
	print 'Exporting wordlist...'
	utils.db.push_table(settings.WORDS_TABLE, settings.FILES['wordlist.db'])
	print 'Exporting surface rules...'
	utils.db.push_table(settings.S_RUL_TABLE, settings.FILES['surface.rules.db'])
	print 'Exporting graph...'
	utils.db.push_table(settings.GRAPH_TABLE, settings.FILES['surface.graph.db'])
	print 'Exporting surface rules co-occurrences...'
	utils.db.push_table(settings.S_RUL_CO_TABLE,\
		settings.FILES['surface.rules.cooc.db'])
	utils.db.close_connection()
	# delete temporary files
	remove_file(settings.FILES['wordlist.db'])
	remove_file(settings.FILES['surface.rules.db'])
	remove_file(settings.FILES['surface.graph.db'])
	remove_file(settings.FILES['surface.rules.cooc.db'])

