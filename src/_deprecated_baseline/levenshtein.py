from algorithms.align import *
from utils.files import *
import settings

TAGS_KNOWN = True
settings.USE_TAGS = True
OUTPUT_FILE = 'levenshtein-kn.txt'

def levdist(word_1, word_2):
	cs = algorithms.align.lcs(word_1, word_2)
	pattern = re.compile('(.*)' + '(.*?)'.join([\
		letter for letter in cs]) + '(.*)')
	dist = 0
	# compare tags
	if settings.USE_TAGS:
		idx_1 = word_1.rfind('_')
		idx_2 = word_2.rfind('_')
		word_1, tag_1 = word_1[:idx_1], word_1[idx_1+1:].split('.')
		word_2, tag_2 = word_2[:idx_2], word_2[idx_2+1:].split('.')
		if TAGS_KNOWN:
			for i in range(max(len(tag_1), len(tag_2))):
				if i >= len(tag_1) or i >= len(tag_2) or tag_1[i] != tag_2[i]:
					dist += 1
	m1 = pattern.search(word_1)
	m2 = pattern.search(word_2)
	for i, (x, y) in enumerate(zip(m1.groups(), m2.groups())):
		dist += max(len(x), len(y))
	return dist

def run():
	# annotate the graph with Levenshtein distances of wordpairs
	outfile = settings.FILES['surface.graph'] + '.lev'
	with open_to_write(outfile) as outfp:
		for word_1, word_2, rule in read_tsv_file(settings.FILES['surface.graph'] + '.orig',\
				print_progress=True, print_msg='Computing Levenshtein distance...'):
			write_line(outfp, (word_1, word_2, rule, levdist(word_1, word_2)))

	sort_file(outfile, outfile+'.so', numeric=True, key=4)
	sort_file(outfile+'.so', outfile+'.so.un', stable=True, unique=True, key=2)

	# evaluate
	lemma = {}
	for word_1, freq, word_2 in read_tsv_file('lexicon.eval'):
		lemma[word_1] = word_2
	correct, total = 0, 0
	with open_to_write(OUTPUT_FILE) as outfp:
		for word_1, word_2, rule in read_tsv_file(outfile+'.so.un', (str, str, str)):
			total += 1
			c = '-'
			if word_1 == lemma[word_2]:
				c = '+'
				correct += 1
			write_line(outfp, (word_2, word_1, lemma[word_2], c))
		write_line(outfp, (correct, total, correct / total))
	print(correct, total, correct/total)

