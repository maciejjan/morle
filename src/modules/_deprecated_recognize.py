from datastruct.lexicon import *
from utils.files import *
import settings

THRESHOLD = 5.0

def run():
	lexicon = Lexicon.load_model(settings.FILES['model.rules'],\
								 settings.FILES['model.lexicon'])
	recognized, total = 0, 0
	for word, freq in read_tsv_file(settings.FILES['testing.wordlist'], (str, int),\
			print_progress=True, print_msg='Recognizing words...'):
		cost = lexicon.try_word(word, 1, 1, ignore_lex_depth=True)[0][3]
#		print(word, cost)
		if cost < THRESHOLD:
			recognized += 1
		total += 1
	print('Recognized: {:.2}\n'.format(recognized/total))

