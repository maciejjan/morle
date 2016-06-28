import numpy as np
import libhfst
import re

ENCODING = 'utf-8'
#SETTINGS_FILE = 'settings.ini'
WORKING_DIR = ''

#MIN_BASE_FREQ = 167
#MIN_RULE_FREQ = 2
ROOTDIST_N = 1
MAX_NUM_RULES = 10000
MAX_AFFIX_LENGTH = 5
MAX_PROD = 0.999
#INDEPENDENCY_THRESHOLD = 0.001
#DERIVATION_THRESHOLD = 0.1

TRAINING_ALGORITHM = 'softem'
SUPERVISED = False
USE_WORD_FREQ = False
WORD_FREQ_WEIGHT = 1.0
WORD_VEC_WEIGHT = 0.0
WORD_VEC_DIM = 100
USE_TAGS = True
#DEBUG_MODE = False
COMPOUNDING_RULES = True
LEMMAS_KNOWN = False

# MCMC SAMPLING
SAMPLING_WARMUP_ITERATIONS = 1000000
SAMPLING_ITERATIONS = 10000000

EM_MAX_ITERATIONS = 10

#TRANSDUCER_TYPE	= libhfst.HFST_OL_TYPE
#TRANSDUCER_TYPE	= libhfst.FOMA_TYPE
TRANSDUCER_TYPE	= libhfst.SFST_TYPE

VECTOR_SEP	= ' '
#TAG_SEP 	= '_'
SYMBOL_PATTERN 			= '(?:\w|\{[A-Z0-9]+\})'
SYMBOL_PATTERN_CMP		= re.compile(SYMBOL_PATTERN)
TAG_PATTERN 			= '(?:<[A-Z0-9]+>)'
TAG_PATTERN_CMP			= re.compile(TAG_PATTERN)
WORD_PATTERN			= '^(?P<word>%s+)(?P<tag>%s*)$' %\
						  (SYMBOL_PATTERN, TAG_PATTERN)
WORD_PATTERN_CMP		= re.compile(WORD_PATTERN)

RULE_SUBST_SEP				= ':'
RULE_PART_SEP				= '/'
RULE_TAG_SEP				= '___'
RULE_SUBST_PATTERN			= '%s*%s%s*' %\
							  (SYMBOL_PATTERN, RULE_SUBST_SEP, SYMBOL_PATTERN)
RULE_NAMED_SUBST_PATTERN	= '(?P<x>%s*)%s(?P<y>%s*)' %\
							  (SYMBOL_PATTERN, RULE_SUBST_SEP, SYMBOL_PATTERN)
RULE_NAMED_SUBST_PATTERN_CMP= re.compile(RULE_NAMED_SUBST_PATTERN)
RULE_TAG_SUBST_PATTERN		= '%s*%s%s*' %\
							  (TAG_PATTERN, RULE_SUBST_SEP, TAG_PATTERN)
RULE_TAG_SUBST_PATTERN_CMP	= re.compile(RULE_SUBST_PATTERN)
RULE_PATTERN				= '^(?P<subst>%s(%s)*)(?:%s(?P<tag_subst>%s))?$' %\
							  (RULE_SUBST_PATTERN,
							   RULE_PART_SEP+RULE_SUBST_PATTERN,
							   RULE_TAG_SEP,
							   RULE_TAG_SUBST_PATTERN)
RULE_PATTERN_CMP		= re.compile(RULE_PATTERN)

RULE_NAMED_TAG_SUBST_PATTERN = '(?P<x>%s*)%s(?P<y>%s*)' %\
							   (TAG_PATTERN, RULE_SUBST_SEP, TAG_PATTERN)
RULE_NAMED_TAG_SUBST_PATTERN_CMP =\
	re.compile(RULE_NAMED_TAG_SUBST_PATTERN)

WORDLIST_FORMAT = (\
	str,\
	int if WORD_FREQ_WEIGHT > 0.0 else None,\
	(lambda x: np.array(list(map(float, x.split(VECTOR_SEP)))))\
	          if WORD_VEC_WEIGHT > 0.0 else None\
)

LEXICON_FORMAT = (\
	str,\
	str,
	str,
	int if WORD_FREQ_WEIGHT > 0.0 else None,\
	(lambda x: np.array(list(map(float, x.split(VECTOR_SEP)))))\
	          if WORD_VEC_WEIGHT > 0.0 else None\
)

# load settings from file or create a settings file
def process_settings_file():
	import os.path
	global FILES
	if os.path.isfile(WORKING_DIR+FILES['settings']):
		print('Loading settings...')
		load_settings(WORKING_DIR+FILES['settings'])
	else:
		print('Saving settings...')
		save_settings(WORKING_DIR+FILES['settings'])

## filenames
FILES = {
	'derivation' : 'derivation.txt',
	'derivation.eval' : 'gs_derivation.txt',
	'derivation.eval.log' : 'derivation.eval.txt',
	'derivation.rules' : 'd_rul.txt',
	'inflection' : 'inflection.txt',
	'inflection.eval' : 'gs_inflection.txt',
	'inflection.eval.log' : 'inflection.eval.txt',
	'inflection.rules' : 'i_rul.txt',
	'inflection.lexemes'	: 'lexemes.txt',
	'inflection.paradigms'	: 'paradigms.txt',
	'surface.graph' : 'graph.txt',
	'surface.graph.partitioned' : 'graph_p.txt',
	'surface.substrings' : 'substrings.txt',
	'surface.rules'	: 'rules_c.txt',
	'surface.rules.cooc'	: 's_rul_co.txt',
	'index' : 'index.txt',
	'segmentation' : 'segmentation.txt',
	'settings' : 'settings.ini',

	# filenames of temporary files while exporting to DB
	'derivation.db' : 'derivation.txt.db',
	'derivation.rules.db' : 'd_rul.txt.db',
	'surface.graph.db' : 'graph.txt.db',
	'inflection.db' : 'inflection.txt.db',
	'inflection.rules.db' : 'i_rul.txt.db',
	'inflection.lexemes.db' : 'lexemes.txt.db',
	'inflection.paradigms.db'	: 'paradigms.txt.db',
	'inflection.par_rul.db'	: 'par_rul.txt.db',
	'surface.rules.db'	: 's_rul.txt.db',
	'surface.rules.cooc.db'	: 's_rul_co.txt.db',
	'trained.rules.db'	: 'tr_rules.txt.db',
	'trained.rules.cooc.db' : 'tr_rules_cooc.txt.db',
	'wordlist.db'	: 'words.txt.db',

	# training data
	'trained.rules' : 'tr_rules.txt',
	'trained.rules.cooc' : 'tr_rules_cooc.txt',
	'training.inflection' : 'tr_infl.txt',
	'training.inflection.graph' : 'tr_infl_graph.txt',
	'training.lexicon' : 'lexicon.training',
	'training.wordlist' : 'input.training',
	'training.substrings' : 'tr_substrings.txt',
	'training.surface.graph' : 'tr_graph.txt',

	# model data
	'model.lexicon' : 'lexicon.txt',
	'model.rules' : 'rules.txt',
	'model.ngrams' : 'unigrams.txt',
	'testing.wordlist' : 'input.testing',
	'analyses' : 'analyses.txt',

	'wordgen.output' : 'wordgen.txt',
	'full.wordlist' : 'wordlist-full.txt',
	'model' : 'model.dat'
}

# DB access data
DB_HOST = 'localhost'
DB_NAME = None
DB_USER = None
DB_PASS = None

# DB table names and create statements
#D_RUL_TABLE = ('d_rul', '''CREATE TABLE `d_rul`(
#	`r_id` int(10) unsigned NOT NULL auto_increment,
#	`s_rul_id` int(10) unsigned NOT NULL default '0',
#	`p1_id` int(10) unsigned default NULL,
#	`p2_id` int(10) unsigned default NULL,
#	`freq` int(10) unsigned default NULL,
#	PRIMARY KEY `r_id` (`r_id`)
#) ENGINE=MyISAM DEFAULT CHARSET=utf8;
#''')
DERIVATION_TABLE = ('derivation', '''CREATE TABLE `derivation`(
	`l1_id` int(10) unsigned NOT NULL default '0',
	`l2_id` int(10) unsigned NOT NULL default '0',
	`r_id` int(10) unsigned NOT NULL default '0',
	PRIMARY KEY `w1_w2` (`w1_id`, `w2_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
GRAPH_TABLE = ('graph', '''CREATE TABLE `graph`(
	`w1_id` int(10) unsigned NOT NULL default '0',
	`w2_id` int(10) unsigned NOT NULL default '0',
	`r_id` int(10) unsigned NOT NULL default '0',
	PRIMARY KEY `w1_w2` (`w1_id`, `w2_id`),
	KEY `r_id` (`r_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
INFLECTION_TABLE = ('inflection', '''CREATE TABLE `inflection`(
	`w_id` int(10) unsigned NOT NULL default '0',
	`l_id` int(10) unsigned NOT NULL default '0',
	`p_id` int(10) unsigned default NULL,
	PRIMARY KEY `w_id` (`w_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
I_RUL_TABLE = ('i_rul', '''CREATE TABLE `i_rul`(
	`r_id` int(10) unsigned NOT NULL auto_increment,
	`rule` varchar(255) character set utf8 collate utf8_bin default NULL,
	`freq` int(10) unsigned default NULL,
	PRIMARY KEY `r_id` (`r_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
LEXEMES_TABLE = ('lexemes', '''CREATE TABLE `lexemes`(
	`l_id` int(10) unsigned NOT NULL auto_increment,
	`lemma` int(10) unsigned NOT NULL default '0',
	`size` int(4) unsigned default NULL,
	PRIMARY KEY `l_id` (`l_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
PARADIGMS_TABLE = ('paradigms', '''CREATE TABLE `paradigms`(
	`p_id` int(10) unsigned NOT NULL auto_increment,
	`freq` int(10) unsigned default NULL,
	`size` int(4) unsigned default NULL,
	`str` varchar(1000) character set utf8 collate utf8_bin default NULL,
	PRIMARY KEY `p_id` (`p_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
PAR_RUL_TABLE = ('par_rul', '''CREATE TABLE `par_rul`(
	`p_id` int(10) unsigned NOT NULL default '0',
	`r_id` int(10) unsigned NOT NULL default '0',
	KEY (`p_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
S_RUL_TABLE = ('s_rul', '''CREATE TABLE `s_rul`(
	`r_id` int(10) unsigned NOT NULL auto_increment,
	`rule` varchar(255) character set utf8 collate utf8_bin default NULL,
	`freq` int(10) unsigned default NULL,
	`prob` float(10) unsigned default NULL,
	PRIMARY KEY `r_id` (`r_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
S_RUL_CO_TABLE = ('s_rul_co', '''CREATE TABLE `s_rul_co`(
	`r1_id` int(10) unsigned NOT NULL default '0',
	`r2_id` int(10) unsigned NOT NULL default '0',
	`freq` int(8) unsigned default NULL,
	`sig` float(8) default NULL,
	PRIMARY KEY `r1_r2` (`r1_id`, `r2_id`),
	KEY `r1_sig` (`r1_id`, `sig`),
	KEY `r2_sig` (`r2_id`, `sig`),
	KEY `sig` (`sig`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
TR_RUL_TABLE = ('tr_rul', '''CREATE TABLE `s_rul`(
	`r_id` int(10) unsigned NOT NULL auto_increment,
	`rule` varchar(255) character set utf8 collate utf8_bin default NULL,
	`ifreq` int(10) unsigned default NULL,
	`freq` int(10) unsigned default NULL,
	`weight` float(10) unsigned default NULL,
	PRIMARY KEY `r_id` (`r_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
TR_RUL_CO_TABLE = ('tr_rul_co', '''CREATE TABLE `tr_s_rul_co`(
	`r1_id` int(10) unsigned NOT NULL default '0',
	`r2_id` int(10) unsigned NOT NULL default '0',
	`freq` int(8) unsigned default NULL,
	`sig` float(8) default NULL,
	PRIMARY KEY `r1_r2` (`r1_id`, `r2_id`),
	KEY `r1_sig` (`r1_id`, `sig`),
	KEY `r2_sig` (`r2_id`, `sig`),
	KEY `sig` (`sig`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')
WORDS_TABLE = ('words', '''CREATE TABLE `words`(
	`w_id` int(10) unsigned NOT NULL auto_increment,
	`word` varchar(255) character set utf8 collate utf8_bin default NULL,
	`freq` int(10) unsigned default NULL,
	PRIMARY KEY `w_id` (`w_id`),
	KEY `word` (`word`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
''')

def load_settings(filename):
	with open(filename) as fp:
		for line in fp:
			if line.strip().startswith(';'):	# comments
				continue
			content = [c.strip() for c in line.strip().split('=')]
			if content[0] == 'MAX_AFFIX_LENGTH':
				global MAX_AFFIX_LENGTH
				MAX_AFFIX_LENGTH = int(content[1])
#			elif content[0] == 'MIN_RULE_FREQ':
#				global MIN_RULE_FREQ
#				MIN_RULE_FREQ = int(content[1])
#			elif content[0] == 'INDEPENDENCY_THRESHOLD':
#				global INDEPENDENCY_THRESHOLD
#				INDEPENDENCY_THRESHOLD = float(content[1])
			elif content[0] == 'MAX_NUM_RULES':
				global MAX_NUM_RULES
				MAX_NUM_RULES = int(content[1])
			elif content[0] == 'ROOTDIST_N':
				global ROOTDIST_N
				ROOTDIST_N = int(content[1])
			elif content[0] == 'SUPERVISED':
				global SUPERVISED
				SUPERVISED = True if content[1] == 'True' else False
			elif content[0] == 'USE_WORD_FREQ':
				global USE_WORD_FREQ
				USE_WORD_FREQ = True if content[1] == 'True' else False
			elif content[0] == 'USE_TAGS':
				global USE_TAGS
				USE_TAGS = True if content[1] == 'True' else False
			elif content[0] == 'LEMMAS_KNOWN':
				global LEMMAS_KNOWN
				LEMMAS_KNOWN = True if content[1] == 'True' else False

def save_settings(filename):
	with open(filename, 'w+') as fp:
		fp.write('MAX_AFFIX_LENGTH = ' + str(MAX_AFFIX_LENGTH) + '\n')
		fp.write('ROOTDIST_N = ' + str(ROOTDIST_N) + '\n')
#		fp.write('INDEPENDENCY_THRESHOLD = ' + str(INDEPENDENCY_THRESHOLD) + '\n')
		fp.write('MAX_NUM_RULES = ' + str(MAX_NUM_RULES) + '\n')
		fp.write('SUPERVISED = ' + str(SUPERVISED) + '\n')
		fp.write('USE_WORD_FREQ = ' + str(USE_WORD_FREQ) + '\n')
		fp.write('USE_TAGS = ' + str(USE_TAGS) + '\n')
		fp.write('LEMMAS_KNOWN = ' + str(LEMMAS_KNOWN) + '\n')
		fp.flush()

