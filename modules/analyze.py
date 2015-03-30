from datastruct.lexicon import *
from utils.files import *
import settings

def add_lemmas_to_lexicon(lexicon, filename):
	for word, freq, base in read_tsv_file(filename, (str, int, str)):
		if not base in lexicon:
			lexicon.add_word(base, 1, lexicon.rootdist.word_prob(base))

def remove_single_roots_from_lexicon(lexicon):
	roots_to_delete = []
	for rt in lexicon.roots:
		if not lexicon[rt].next:
			roots_to_delete.append(rt)
	for rt in roots_to_delete:
		lexicon.roots.remove(rt)
		del lexicon[rt]

def analyze_word(word, lexicon, max_results=1):
	'''Return tagged word and lemma for a given word.'''

	def find_word_in_subtree(word, subtree):
		results = []
		for node in subtree.search():
			if node.key() == word or (node.word == word and word.rfind('_') == -1):
				base_node = node.prev if node.prev is not None else node
				results.append((node.key(), base_node.key()))
		return results
	
	results = []
	for base, rule, subtree, cost in lexicon.try_word(word, 1, max_results=max_results):
		if base is not None:
			results.append((subtree.key(), base))
		else:
			results.extend(find_word_in_subtree(word, subtree))
	return results[:max_results]

def analyze(filename, lexicon, outfile=None, known_tags=False, max_results=1):

	def compare_pair(x, y):
		return (x[0] == y[0], x[1] == y[1])

	correct_tag, correct_base, correct, total = 0, 0, 0, 0
	for word, freq, base in read_tsv_file(filename, (str, int, str),\
#			):
			print_progress=True, print_msg='Analyzing...'):
		word_to_analyze = word if known_tags else word[:word.rfind('_')]
		analyses = analyze_word(word_to_analyze, lexicon, max_results)
		best_analysis = max(analyses, key = lambda x: compare_pair(x, (word, base)))
		score = compare_pair(best_analysis, (word, base))
		correct += 1 if score == (True, True) else 0
		correct_tag += 1 if score[0] else 0
		correct_base += 1 if score[1] else 0
		c_tag = '+' if score[0] else '-'
		c_base = '+' if score[1] else '-'
		total += 1
		if outfile:
			outfile.write('\t'.join([best_analysis[0], best_analysis[1], word, base, c_base, c_tag]))
			outfile.write('\n')

	return correct_base / total, correct_tag / total, correct / total

def run():
	lexicon = Lexicon.load_model(settings.FILES['model.rules'],\
								 settings.FILES['model.lexicon'])
#	lexicon.nodes = {}
	if not settings.SUPERVISED:
		remove_single_roots_from_lexicon(lexicon)
	if settings.LEMMAS_KNOWN:
		add_lemmas_to_lexicon(lexicon, 'lexicon.eval')
	with open_to_write('eval-kn.txt') as outfp:
		c_b, c_t, c = analyze('lexicon.eval', lexicon, outfp, True)
		outfp.write('=== KNOWN TAGS ===\n\n')
		outfp.write('Lemmas: {:.2}\n'.format(c_b))
		outfp.write('Tags: {:.2}\n'.format(c_t))
		outfp.write('Lemmas+Tags: {:.2}\n'.format(c))
	with open_to_write('eval-unkn-top1.txt') as outfp:
		c_b, c_t, c = analyze('lexicon.eval', lexicon, outfp, False)
		outfp.write('=== UNKNOWN TAGS ===\n\n')
		outfp.write('Lemmas: {:.2}\n'.format(c_b))
		outfp.write('Tags: {:.2}\n'.format(c_t))
		outfp.write('Lemmas+Tags: {:.2}\n'.format(c))
#	with open_to_write('eval-unkn-top3.txt') as outfp:
#		c_b, c_t, c = analyze('lexicon.eval', lexicon, outfp, False, 3)
#		outfp.write('=== UNKNOWN TAGS (TOP-3) ===\n\n')
#		outfp.write('Lemmas: {:.2}\n'.format(c_b))
#		outfp.write('Tags: {:.2}\n'.format(c_t))
#		outfp.write('Lemmas+Tags: {:.2}\n'.format(c))
#	with open_to_write('eval-unkn-top5.txt') as outfp:
#		c_b, c_t, c = analyze('lexicon.eval', lexicon, outfp, False, 5)
#		outfp.write('=== UNKNOWN TAGS (TOP-5) ===\n\n')
#		outfp.write('Lemmas: {:.2}\n'.format(c_b))
#		outfp.write('Tags: {:.2}\n'.format(c_t))
#		outfp.write('Lemmas+Tags: {:.2}\n'.format(c))

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
