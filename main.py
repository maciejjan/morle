from modules import *
import settings
import argparse

def setup():
	'Process command-line parameters and update settings.'
	ap = argparse.ArgumentParser(prog='gramm', \
		description='GRAMM -- GRaph-based A-Morphous Morphologizer 1.0')
	# obligatory arguments
	ap.add_argument('mode', type=str, help='import, export, eval, run')
	ap.add_argument('modules', type=str, help='srules, infl, deriv')
	# optional arguments
	ap.add_argument('-d', action='store', dest='workdir', help='working directory')
	ap.add_argument('-s', action='store_true', dest='supervised', help='supervised learning')
	ap.add_argument('-t', action='store_true', dest='use_tags', help='use POS-tags')
#	ap.add_argument('-f', '--force', action='store_true', help='force')
	ap.add_argument('--db-host', type=str, action='store', dest='db_host',\
		help='database host (default: localhost) (import/export mode)')
	ap.add_argument('--db-user', type=str, action='store', dest='db_user',\
		help='database username (import/export mode)')
	ap.add_argument('--db-pass', type=str, action='store', dest='db_pass',\
		help='database password (import/export mode)')
	ap.add_argument('--db-name', type=str, action='store', dest='db_name',\
		help='database name (import/export mode)')
	ap.add_argument('--version', action='version', version='GRAMM 1.0')
#	ap.add_argument('-p', '--progress', action='store_true', help='print progress of performed operations')
	args = ap.parse_args()
	if args.workdir is not None:
		if not args.workdir.endswith('/'):
			args.workdir += '/'
		settings.WORKING_DIR = args.workdir
	if args.supervised:
		settings.SUPERVISED = True
	if args.use_tags:
		settings.USE_TAGS = True
	if args.db_host is not None:
		settings.DB_HOST = args.db_host
	if args.db_user is not None:
		settings.DB_USER = args.db_user
	if args.db_pass is not None:
		settings.DB_PASS = args.db_pass
	if args.db_name is not None:
		settings.DB_NAME = args.db_name
#	settings.setup_filenames()
	settings.process_settings_file()
	return args.mode.split('+'), args.modules.split('+')

def main(mode, modules):

	MODE_IMPORT = 'import'
	MODE_EXPORT = 'export'
	MODE_EVAL = 'eval'
	MODE_RUN = 'run'

	MODULE_PRE = 'pre'
#	MODULE_LEXEMES = 'infl'
	MODULE_TRAIN = 'train'
	MODULE_MCMC = 'mcmc'
	MODULE_ANALYZE = 'analyze'
#	MODULE_TAG = 'tag'
#	MODULE_DERIV = 'deriv'

	if MODE_IMPORT in mode:
		if MODULE_PRE in modules:
			preprocess.import_from_db()
#		if MODULE_LEXEMES in modules:
#			lexemes.import_from_db()
#		if MODULE_DERIV in modules:
#			derivation.import_from_db()
		if MODULE_TRAIN in modules:
			train.import_from_db()
	if MODE_RUN in mode:
		if MODULE_PRE in modules:
			preprocess.run()
#		if MODULE_LEXEMES in modules:
#			lexemes.run()
#		if MODULE_DERIV in modules:
#			derivation.run()
		if MODULE_TRAIN in modules:
			train.run()
		if MODULE_MCMC in modules:
			mcmc.run()
		if MODULE_ANALYZE in modules:
			analyze.run()
		if 'infl' in modules:
			inflect.run()
		if 'recognize' in modules:
			recognize.run()
		if 'wordgen' in modules:
			wordgen.run()
#		if MODULE_TAG in modules:
#			tag.run()
	if MODE_EVAL in mode:
		if MODULE_PRE in modules:
			preprocess.evaluate()
#		if MODULE_LEXEMES in modules:
#			lexemes.evaluate()
#		if MODULE_DERIV in modules:
#			derivation.evaluate()
		if MODULE_TRAIN in modules:
			train.evaluate()
		if MODULE_ANALYZE in modules:
			analyze.evaluate()
	if MODE_EXPORT in mode:
		if MODULE_PRE in modules:
			preprocess.export_to_db()
#		if MODULE_LEXEMES in modules:
#			lexemes.export_to_db()
#		if MODULE_DERIV in modules:
#			derivation.export_to_db()
		if MODULE_TRAIN in modules:
			train.export_to_db()

if __name__ == '__main__':
	mode, modules = setup()
	main(mode, modules)

