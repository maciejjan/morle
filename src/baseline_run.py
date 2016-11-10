import sys
import settings

MODULE = sys.argv[1]
settings.WORKING_DIR = sys.argv[2]
if not settings.WORKING_DIR.endswith('/'):
	settings.WORKING_DIR += '/'

if MODULE == 'lev':
	import baseline.levenshtein
	baseline.levenshtein.run()
elif MODULE == 'maxent':
	import baseline.maxent
	baseline.maxent.run()
elif MODULE == 'maxent_infl':
	import baseline.maxent_infl
	baseline.maxent_infl.run()
elif MODULE == 'morphisto':
	import baseline.wordgen_morphisto
	baseline.wordgen_morphisto.run()
