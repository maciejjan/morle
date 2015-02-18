ENCODING = 'utf-8'
#SETTINGS_FILE = 'settings.ini'
WORKING_DIR = ''

#MIN_BASE_FREQ = 167
#MIN_RULE_FREQ = 2
MAX_NUM_RULES = 10000
MAX_AFFIX_LENGTH = 5
MAX_PROD = 0.999
INDEPENDENCY_THRESHOLD = 0.001
DERIVATION_THRESHOLD = 0.1

SUPERVISED = False
USE_WORD_FREQ = True
USE_TAGS = True
DEBUG_MODE = False
COMPOUNDING_RULES = True

# load settings from file or create a settings file
def process_settings_file():
	import os.path
	global FILES
	if os.path.isfile(FILES['settings']):
		load_settings(FILES['settings'])
	else:
		save_settings(FILES['settings'])

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
	'training.lexicon' : 'input.training',
	'training.wordlist' : 'input.training',
	'training.substrings' : 'tr_substrings.txt',
	'training.surface.graph' : 'tr_graph.txt',

	# model data
	'model.lexicon' : 'lexicon.txt',
	'model.rules' : 'rules.txt',
	'model.ngrams' : 'unigrams.txt',
	'testing.wordlist' : 'input.testing',
	'analyses' : 'analyses.txt'
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
	with open(WORKING_DIR+filename) as fp:
		for line in fp:
			if line.strip().startswith(';'):
				continue
			content = [c.strip() for c in line.strip().split('=')]
			if content[0] == 'MAX_AFFIX_LENGTH':
				global MAX_AFFIX_LENGTH
				MAX_AFFIX_LENGTH = int(content[1])
			elif content[0] == 'MIN_RULE_FREQ':
				global MIN_RULE_FREQ
				MIN_RULE_FREQ = int(content[1])
			elif content[0] == 'INDEPENDENCY_THRESHOLD':
				global INDEPENDENCY_THRESHOLD
				INDEPENDENCY_THRESHOLD = float(content[1])

def save_settings(filename):
	with open(WORKING_DIR+filename, 'w+') as fp:
		fp.write('MAX_AFFIX_LENGTH = ' + str(MAX_AFFIX_LENGTH) + '\n')
		fp.write('INDEPENDENCY_THRESHOLD = ' + str(INDEPENDENCY_THRESHOLD) + '\n')
		fp.flush()

