from datastruct.rules import *
from datastruct.lexicon import *
from algorithms.ngrams import *
from utils.files import *
import math

RULES_FILE = 'rules.txt'
LEXICON_FILE = 'lexicon.txt'
EVAL_LEXICON_FILE = 'lexicon.eval'
ANALYSES_FILE = 'analyses.txt'
NGRAM_MODEL_FILE = 'unigrams.txt'
EVALUATION_FILE = 'eval.txt'

# TODO take 5 best analyses and try to analyse them further
# word frequencies on/off
# TODO conversion of only tags, with word staying the same (e.g. VVINF->VVFIN)

def rule_score(ruledata, freq, unigrams):
	rule = Rule.from_string(ruledata.rule)
	score = math.log(unigrams.word_prob(rule.prefix[0])) - math.log(unigrams.word_prob(rule.prefix[1]))
	for (x, y) in rule.alternations:
		score += math.log(unigrams.word_prob(x)) - math.log(unigrams.word_prob(y))
	score += math.log(unigrams.word_prob(rule.suffix[0])) - math.log(unigrams.word_prob(rule.suffix[1]))
	score += math.log(ruledata.prod)
	score += freq * math.log(ruledata.weight / (1.0 + ruledata.weight))
	return score

def analyze_word(word, freq, unigrams, rules, lexicon):
	if lexicon.has_key(word) and lexicon[word].training:
		if lexicon[word].prev is not None:
			return lexicon[word].prev.word
		else:
			return None
	ngr_prob = unigrams.word_prob(word)
	lexicon.add_word(word, freq, ngr_prob)
	for r in rules.values():
		if r.prod == 1.0:
			r.prod = 0.999
#	print word, ngr_prob
	rules_list = [(r.rule, r.prod / (ngr_prob * (1.0 - r.prod))) for r in rules.values() if r.rule != u'#']
	rules_list.sort(reverse = True, key = lambda x:x[1])
	max_imp, max_imp_word = 0.0, None
	max_imp_rule = None
#	print word.encode('utf-8')
	for r, imp in rules_list:
		if imp < max_imp:
			break
		irule = Rule.from_string(r).reverse()
		if irule.lmatch(word):
			for w in irule.apply(word):
				if w == word:
					break
				score = lexicon.try_edge(w, word, rules[r]) if lexicon.has_key(w)\
					else rule_score(rules[r], freq, unigrams)
#				print irule.to_string().encode('utf-8'), score, lexicon.has_key(w)
				if score > max_imp and\
						not (lexicon.has_key(w) and word in lexicon[w].analysis()):
					max_imp = score
					max_imp_word = w
					max_imp_rule = r
#	print ''
	if max_imp_word is not None:
		if not lexicon.has_key(max_imp_word):
			lexicon.add_word(max_imp_word, 0.001, unigrams.word_prob(max_imp_word))
		lexicon.draw_edge(max_imp_word, word, rules[max_imp_rule])
	return max_imp_word if max_imp_word else None

def analyze_wordlist(input_file, output_file, unigrams, rules, lexicon):
	# add all words from the testing data to lexicon
	added_freq = 0
	for word, freq in read_tsv_file(input_file, (unicode, int), True):
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
	unigrams = NGramModel.load_from_file(NGRAM_MODEL_FILE)
#	print analyze_word(u'schwest', unigrams, rules, lexicon)
	analyze_wordlist(settings.FILES['testing.wordlist'], ANALYSES_FILE, unigrams, rules, lexicon)

def evaluate_direct():
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
				total += 1
				if stems_gs[word] == stem:
					write_line(outfp, (word, '+', stem))
					correct += 1
				else:
					write_line(outfp, (word, '-', stem, stems_gs[word]))
	print 'Correctness: %0.2f (%d/%d)' %\
		(float(correct) * 100 / total, correct, total)

def evaluate():
	evaluate_direct()

