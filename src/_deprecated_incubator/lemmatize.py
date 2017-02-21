from utils.files import *
import sys

SENTENCES_FILE = sys.argv[1]
INV_W_FILE = sys.argv[2]
DICTIONARY_FILE = sys.argv[3]

def load_dictionary(filename):
	dictionary = {}
	for row in load_tsv_file(filename):
		dictionary[row[0]] = row[1]
	return dictionary

dictionary = load_dictionary(DICTIONARY_FILE)
s_file = load_tsv_file(SENTENCES_FILE)
for i_s_id, i_words in load_tsv_file_by_key(INV_W_FILE, 2):
	row = s_file.next()
	if len(row) >= 2:
		s_id, sentence = row[0], row[1]
		while int(i_s_id) > int(s_id):
			print s_id.encode('utf-8') + '\t' + sentence.encode('utf-8')
			row = s_file.next()
			if len(row) < 2:
				continue
			s_id, sentence = row[0], row[1]
		for (word, ) in i_words:
			if dictionary.has_key(word.lower()):
				sentence = sentence.replace(' '+word, ' '+dictionary[word.lower()])
				sentence = sentence.replace(word+' ', dictionary[word.lower()]+' ')
		print s_id.encode('utf-8') + '\t' + sentence.encode('utf-8')

