from datastruct.lexicon import *
from utils.files import *
import settings

def add_lemmas_to_lexicon(lexicon, filename):
	for word, freq, base in read_tsv_file(filename, (str, int, str)):
		if not base in lexicon:
			lexicon.add_word(base, 1, lexicon.rootdist.word_prob(base))

def inflect_word(word, tag, lexicon):
	# if the tag is identical with that of lemma -> return the lemma
	if word.endswith(tag):
		return word
	# if the word in question is in the lexicon -> return it
	for w in lexicon[word].next.values():
		if w.tag == tag:
			return w.key()
	# otherwise use the best rule
	for r in sorted(lexicon.ruleset.values(), reverse=True, key=lambda r: r.prod):
		if r.rule_obj.tag[1] == tag and r.rule_obj.lmatch(word):
			a = r.rule_obj.apply(word)
			if a:
				return a[0]
	return None

def inflect(filename, lexicon, outfile=None):
	correct, total = 0, 0
	for word, freq, base in read_tsv_file(filename, (str, int, str),\
#			):
			print_progress=True, print_msg='Analyzing...'):
		tag = word[word.rfind('_')+1:]
		word_inf = inflect_word(base, tag, lexicon)
		c = '+' if word_inf == word else '-'
		correct += 1 if c == '+' else 0
		total += 1
		if outfile:
			outfile.write('\t'.join([base, tag, str(word_inf), word, c]))
			outfile.write('\n')

	return correct / total

def run():
	lexicon = Lexicon.load_model(settings.FILES['model.rules'],\
								 settings.FILES['model.lexicon'])
#	lexicon.nodes = {}
#	remove_single_roots_from_lexicon(lexicon)
	add_lemmas_to_lexicon(lexicon, 'lexicon.eval')
	with open_to_write('eval-infl.txt') as outfp:
		c = inflect('lexicon.eval', lexicon, outfp)
		outfp.write('Correct: {:.2}\n'.format(c))
		print('Correct: {:.2}\n'.format(c))

def evaluate():
	pass

#def evaluate_direct():
#	stems_gs = {}
#	for word, freq, stem in read_tsv_file(EVAL_LEXICON_FILE):
#		if stem == u'-' or stem == word:
#			stems_gs[word] = None
#		else:
#			stems_gs[word] = stem
#	total, correct = 0, 0
#	with open_to_write(EVALUATION_FILE) as outfp:
#		for word, freq, stem in read_tsv_file(ANALYSES_FILE):
#			if stem == 'None':
#				stem = None
#			if stems_gs.has_key(word):
#				total += 1
#				if stems_gs[word] == stem:
#					write_line(outfp, (word, '+', stem))
#					correct += 1
#				else:
#					write_line(outfp, (word, '-', stem, stems_gs[word]))
#	print('Correctness: %0.2f (%d/%d)' %\
#		(float(correct) * 100 / total, correct, total))
#
#def evaluate():
#	evaluate_direct()
#
