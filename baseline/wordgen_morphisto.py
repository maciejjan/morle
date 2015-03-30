# evaluate word generation using morphisto

from utils.files import *
import subprocess
import sys

WORKING_DIR = sys.argv[2]
if not WORKING_DIR.endswith('/'):
	WORKING_DIR += '/'
INPUT_FILE = WORKING_DIR + 'wordgen.txt'
OUTPUT_FILE = WORKING_DIR + 'wordgen.txt.eval'
MORPHISTO_CMD = ['fst-infl', '/home/mjanicki/software/morphisto/morphisto-02022011.a']

def evaluate(input_file, output_file):
	m_in = ''
	for word, cost, c in read_tsv_file(INPUT_FILE, (str, float, str),\
			print_progress=True, print_msg='Reading input...'):
		m_in += word+'\n'
		if len(word) > 1:
			m_in += word[0].upper() + word[1:] + '\n'
	p = subprocess.Popen(MORPHISTO_CMD, universal_newlines=True,\
		stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	m_out, m_err = p.communicate(m_in)
	cur_word, found = None, False
	with open_to_write(output_file) as outfp:
		for line in m_out.split('\n'):
			if line.startswith('>'):
				new_word = line.rstrip()[2:].lower()
				if new_word != cur_word:
					if cur_word:
						write_line(outfp, (cur_word, ('+' if found else '-')))
#					print(cur_word + '\t' + ('+' if found else '-'))
					cur_word = new_word
					found = False
			elif not line or line.startswith('no result for'):
				pass
			else:
				found = True
		if cur_word:
#		print(cur_word + '\t' + ('+' if found else '-'))
			write_line(outfp, (cur_word, ('+' if found else '-')))

def run():
	evaluate(INPUT_FILE, OUTPUT_FILE)

