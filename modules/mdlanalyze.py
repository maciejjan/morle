from datastruct.rules import *
from datastruct.lexicon import *
from algorithms.ngrams import *
from utils.files import *
import settings
import math
import sys

INPUT_FILE = 'input.testing'
RULES_FILE = 'rules.txt'
LEXICON_FILE = 'lexicon.txt'
EVAL_LEXICON_FILE = 'lexicon.eval'
ANALYSES_FILE = 'analyses.txt'
NGRAM_MODEL_FILE = 'unigrams.txt'
EVALUATION_FILE = 'eval.txt'

# TODO graph search
# TODO conversion of only tags, with word staying the same (e.g. VVINF->VVFIN)

def rule_score(ruledata, unigrams):
	rule = Rule.from_string(ruledata.rule)
	score = math.log(unigrams.word_prob(rule.prefix[0])) - math.log(unigrams.word_prob(rule.prefix[1]))
	for (x, y) in rule.alternations:
		score += math.log(unigrams.word_prob(x)) - math.log(unigrams.word_prob(y))
	score += math.log(unigrams.word_prob(rule.suffix[0])) - math.log(unigrams.word_prob(rule.suffix[1]))
	score += math.log(ruledata.prod)
	return score

def analyze_word(word, freqcl, unigrams, rules, lexicon):
	if lexicon.has_key(word) and lexicon[word].training:
		if lexicon[word].prev is not None:
			return lexicon[word].prev.word
		else:
			return None
#	ngr_prob = unigrams.word_prob(word)
#	lexicon.add_word(word, freqcl, ngr_prob)
	for r in rules.values():
		if r.prod == 1.0:
			r.prod = 0.999
#	print word, ngr_prob
	rules_list = [(r.rule, math.log(r.prod / (lexicon[word].ngram_prob * (1.0 - r.prod)))) for r in rules.values() if r.rule != u'#']
	rules_list.sort(reverse = True, key = lambda x:x[1])
	max_imp, max_imp_word = 0.0, None
	max_imp_rule = None
	if settings.DEBUG_MODE:
		print word.encode('utf-8')
	for r, imp in rules_list:
		if imp < max_imp:
			break
		irule = Rule.from_string(r).reverse()
		if irule.lmatch(word):
#			if settings.DEBUG_MODE:
#				print irule.to_string().encode('utf-8'), imp
			for w in irule.apply(word):
				if w == word:
					break
				score = lexicon.try_edge(w, word, rules[r]) if lexicon.has_key(w)\
					else rule_score(rules[r], unigrams)
				if rules.has_key(irule.to_string()) and lexicon.has_key(w)\
					and score < lexicon.try_edge(word, w, rules[irule.to_string()]):
					continue
				score = lexicon.try_edge(w, word, rules[r]) if lexicon.has_key(w)\
					else rule_score(rules[r], unigrams)
				if settings.DEBUG_MODE and lexicon.has_key(w):
#					print '-', irule.to_string().encode('utf-8'),\
					print '-', r.encode('utf-8'),\
						w.encode('utf-8'), score, lexicon.has_key(w)
				if score > max_imp and lexicon.has_key(w) and\
						not (lexicon.has_key(w) and word in lexicon[w].analysis()):
					max_imp = score
					max_imp_word = w
					max_imp_rule = r
	if settings.DEBUG_MODE:
		print ''
	if max_imp_word is not None:
		if not lexicon.has_key(max_imp_word):
			lexicon.add_word(max_imp_word, 0.001, unigrams.word_prob(max_imp_word))
		lexicon.draw_edge(max_imp_word, word, rules[max_imp_rule])
	return max_imp_word if max_imp_word else None

def analyze_wordlist(input_file, output_file, unigrams, rules, lexicon):
	# add all words from the testing data to lexicon
	for word, freq in read_tsv_file(input_file, (unicode, int), True):
		if not lexicon.has_key(word):
			ngr_prob = unigrams.word_prob(word)
			lexicon.add_word(word, freq, ngr_prob)
			lexicon[word].training = False
	# analyze words
	with open_to_write(output_file) as outfp:
		for word, freq in read_tsv_file(input_file, (unicode, int),\
				print_progress=True, print_msg='Analyzing words...'):
			w = analyze_word(word, freq, unigrams, rules, lexicon)
			write_line(outfp, (word, freq, w))

def analyze_from_stdin(input_file, unigrams, rules, lexicon):
	# add all words from the testing data to lexicon
	for word, freq in read_tsv_file(input_file, (unicode, int), True):
			ngr_prob = unigrams.word_prob(word)
			lexicon.add_word(word, freq, ngr_prob)
			lexicon[word].training = False
	for line in sys.stdin:
		word = line.rstrip().decode('utf-8')
		if lexicon.has_key(word):
			w = analyze_word(word, lexicon[word].freq, unigrams, rules, lexicon)
			if w is not None:
				print w.encode('utf-8')
			else:
				print w
		else:
			print 'Not found.'

def analyze_with_lemmas(input_file, output_file, unigrams, rules, lexicon):
	# add all words from the testing data to lexicon
	added_freq = 0
#	for word, freq in read_tsv_file(input_file, (unicode, int), True):
	word_freq = {}
	for word, freq in read_tsv_file(input_file, (unicode, int), True):
		word_freq[word] = freq
	for w, freq, word in read_tsv_file(EVAL_LEXICON_FILE, (unicode, int, unicode), True):
		if word_freq.has_key(word):
			freq = word_freq[word]
		else:
			freq = 1
		if not lexicon.has_key(word):
			ngr_prob = unigrams.word_prob(word)
			lexicon[word] = LexiconNode(word, freq, freq, ngr_prob, 0.0, 1.0)
			lexicon[word].training = False
			lexicon.roots.add(word)
			lexicon.total += freq
			added_freq += freq
	for rt in lexicon.roots:
		if lexicon[rt].corpus_prob > 0.0:
			lexicon[rt].forward_multiply_corpus_prob(float(lexicon.total-added_freq) / lexicon.total)
		else:
			lexicon[rt].corpus_prob = float(lexicon[rt].freq) / lexicon.total
	# analyze words
	with open_to_write(output_file) as outfp:
		for word, freq in read_tsv_file(input_file, (unicode, int),\
				print_progress=True, print_msg='Analyzing words...'):
			w = analyze_word(word, freq, unigrams, rules, lexicon)
			write_line(outfp, (word, freq, w))

def run():
	rules = RuleSet.load_from_file(RULES_FILE)
#	lexicon = Lexicon()
	lexicon = Lexicon.load_from_file(LEXICON_FILE)
#	unigrams = NGramModel.load_from_file(NGRAM_MODEL_FILE)
	unigrams = NGramModel(1)
	unigrams.train_from_file(INPUT_FILE)
#	print analyze_word(u'schwest', unigrams, rules, lexicon)
	if settings.DEBUG_MODE:
		analyze_from_stdin(settings.FILES['testing.wordlist'], unigrams, rules, lexicon)
	else:
#		analyze_with_lemmas(settings.FILES['testing.wordlist'], ANALYSES_FILE, unigrams, rules, lexicon)
		analyze_wordlist(settings.FILES['testing.wordlist'], ANALYSES_FILE, unigrams, rules, lexicon)

def evaluate_direct():
	unigrams = NGramModel(1)
	unigrams.train_from_file(INPUT_FILE)
	lexicon = Lexicon.load_from_file(LEXICON_FILE)
	stems_gs = {}
	for word, freq, stem in read_tsv_file(EVAL_LEXICON_FILE):
		if stem == u'-' or stem == word:
			stems_gs[word] = None
		else:
			stems_gs[word] = stem
	total, correct = 0, 0
	with open_to_write(EVALUATION_FILE) as outfp:
		for word, freq, stem in read_tsv_file(ANALYSES_FILE):
			if stem == 'None':
				stem = None
			if stems_gs.has_key(word):
				known = 'kn' if (word is None or lexicon.has_key(stems_gs[word])) else 'unkn'
				total += 1
				if stems_gs[word] == stem:
					write_line(outfp, (word, '+', stem, known))
					correct += 1
				else:
					write_line(outfp, (word, '-', stem, stems_gs[word], known))
	print 'Correctness: %0.2f (%d/%d)' %\
		(float(correct) * 100 / total, correct, total)

def evaluate():
	evaluate_direct()

