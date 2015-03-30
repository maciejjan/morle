from datastruct.lexicon import *
from utils.files import *
import settings

def run():
	lexicon = Lexicon.load_model(settings.FILES['model.rules'],\
								 settings.FILES['model.lexicon'])
	word_set = set()
	if file_exists(settings.FILES['full.wordlist']):
		for (word, ) in read_tsv_file(settings.FILES['full.wordlist'], (str, ),\
				print_progress=True, print_msg='Loading dictionary...'):
			word_set.add(word.lower())
	print('Generating words...')
	with open_to_write(settings.FILES['wordgen.output']) as outfp:
		for word, cost in lexicon.expand(max_cost=-5.0):
			idx = word.rfind('_')
			word_notag = word if idx == -1 else word[:idx]
			e = '+' if word_notag in word_set else '-'
			write_line(outfp, (word, cost, e))

