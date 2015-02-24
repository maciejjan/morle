from datastruct.lexicon import *
from utils.files import *
import settings

#TODO
#def analyze_word()
# transform results of try_word() to word-base-tag

def analyze(filename, lexicon, known_tags=False):
	correct_tag, correct_base, correct, total = 0, 0, 0, 0
	for word, freq, base in read_tsv_file(filename, (str, int, str),\
			):
#			print_progress=True, print_msg='Analyzing...'):
		idx = word.rfind('_')
		word, tag = word[:idx], word[idx+1:]
		p_base, p_rule, p_subtree, p_cost = lexicon.try_word(\
				(word+'_'+tag if known_tags else word), 1)
		p_word = None
		if p_base is None:
			p_base = p_subtree.word
		if p_subtree.next and p_cost != 0:
			p_word = p_subtree.next[list(p_subtree.next.keys())[0]].word
		else:
			p_word = p_subtree.word
		p_tag = p_word[p_word.rfind('_')+1:]
		if p_base == base and p_tag == tag:
			correct += 1
			correct_base += 1
			correct_tag += 1
		elif p_base[:p_base.rfind('_')] == base[:base.rfind('_')]:
			correct_base += 1
		elif p_tag == tag:
			correct_tag += 1
		total += 1
		c_base = '+' if p_base == base else '-'
		c_tag = '+' if p_tag == tag else '-'
		print('\t'.join([p_word, p_base, word+'_'+tag, base, c_base, c_tag]))
	return correct_base / total, correct_tag / total, correct / total

def run():
	lexicon = Lexicon.load_model(settings.FILES['model.rules'],\
								 settings.FILES['model.lexicon'])
	lexicon.nodes = {}
	c_b, c_t, c = analyze('lexicon.eval', lexicon, False)
	print('=== UNKNOWN TAGS ===')
	print()
	print('Lemmas: {:.2}'.format(c_b))
	print('Tags: {:.2}'.format(c_t))
	print('Lemmas+Tags: {:.2}'.format(c))
	c_b, c_t, c = analyze('lexicon.eval', lexicon, True)
	print('=== KNOWN TAGS ===')
	print()
	print('Lemmas: {:.2}'.format(c_b))
	print('Tags: {:.2}'.format(c_t))
	print('Lemmas+Tags: {:.2}'.format(c))

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
