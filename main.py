import shared
import argparse
import configparser
import logging
import os.path
import re
#import warnings

def process_config():
    '''Load configuration from the working directory. If missing, load
       the default configuration and save it in the working directory.'''

    def load_config(path):
        shared.config = configparser.ConfigParser()
        shared.config.read(path)

    def save_config(path):
        with open(path, 'w+') as fp:
            shared.config.write(fp)

    config_file_path = os.path.join(
        shared.options['working_dir'], shared.filenames['config'])
    if os.path.isfile(config_file_path):
        load_config(config_file_path)
    else:
        default_config_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            shared.filenames['config-default'])
        if not os.path.isfile(default_config_file_path):
            raise RuntimeError('No configuration file found.')
        load_config(default_config_file_path)
        save_config(config_file_path)

def setup_logger(quiet, verbose):
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    console_formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING if quiet else logging.INFO)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(lambda x: not x.msg.endswith('% done'))
    logger.addHandler(console_handler)

    logfile = os.path.join(
        shared.options['working_dir'],
        shared.filenames['log'])
    encoding = shared.config['General'].get('encoding')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt=shared.config['General'].get('date_format'))

    file_handler = logging.FileHandler(logfile, 'w+', encoding=encoding)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(lambda x: re.match('.*[1-9]% done', x.msg) is None)
    logger.addHandler(file_handler)

def setup():
    'Process command-line parameters and update config.'
    ap = argparse.ArgumentParser(prog='gramm', \
        description='GRAMM -- GRaph-based A-Morphous Morphologizer 1.0')
    # obligatory arguments
    ap.add_argument('mode', type=str, help='import, export, eval, run')
    ap.add_argument('modules', type=str, help='srules, infl, deriv')
    # optional arguments
    ap.add_argument('-d', action='store', dest='workdir', help='working directory')
    ap.add_argument('-s', action='store_true', dest='supervised', help='supervised learning')
    ap.add_argument('-t', action='store_true', dest='use_tags', help='use POS-tags')
    ap.add_argument('-q', action='store_true', dest='quiet',
                          help='quiet mode: print less console output')
    ap.add_argument('-v', action='store_true', dest='verbose',
                          help='verbose mode: print more information in the log file')
#    ap.add_argument('-f', '--force', action='store_true', help='force')
#    ap.add_argument('--db-host', type=str, action='store', dest='db_host',\
#        help='database host (default: localhost) (import/export mode)')
#    ap.add_argument('--db-user', type=str, action='store', dest='db_user',\
#        help='database username (import/export mode)')
#    ap.add_argument('--db-pass', type=str, action='store', dest='db_pass',\
#        help='database password (import/export mode)')
#    ap.add_argument('--db-name', type=str, action='store', dest='db_name',\
#        help='database name (import/export mode)')
    ap.add_argument('--version', action='version', version='GRAMM 0.3.1 alpha')
#    ap.add_argument('-p', '--progress', action='store_true', help='print progress of performed operations')
    args = ap.parse_args()
    if args.workdir is not None:
        shared.options['working_dir'] = os.path.normpath(args.workdir)
    if not os.path.isdir(shared.options['working_dir']):
        raise RuntimeError('%s: the supplied working directory does not exist!' %\
                           shared.options['working_dir'])
    shared.options['quiet'] = args.quiet
    shared.options['verbose'] = args.verbose
    process_config()
    setup_logger(args.quiet, args.verbose)
    return args.mode.split('+'), args.modules.split('+')

def main(mode, modules_to_run):

    MODE_IMPORT = 'import'
    MODE_EXPORT = 'export'
    MODE_EVAL = 'eval'
    MODE_RUN = 'run'

    MODULE_PRE = 'pre'
#    MODULE_LEXEMES = 'infl'
    MODULE_TRAIN = 'train'
    MODULE_MCMC = 'mcmc'
    MODULE_ANALYZE = 'analyze'
#    MODULE_TAG = 'tag'
#    MODULE_DERIV = 'deriv'

#    if MODE_IMPORT in mode:
#        if MODULE_PRE in modules_to_run:
#            preprocess.import_from_db()
##        if MODULE_LEXEMES in modules:
##            lexemes.import_from_db()
##        if MODULE_DERIV in modules:
##            derivation.import_from_db()
#        if MODULE_TRAIN in modules_to_run:
#            train.import_from_db()
    if MODE_RUN in mode:
        if MODULE_PRE in modules_to_run:
            import modules.preprocess
            modules.preprocess.run()
        if 'modsel' in modules_to_run:
            import modules.modsel
            modules.modsel.run()
        if 'fit' in modules_to_run:
            import modules.fit
            modules.fit.run()
        if 'sample' in modules_to_run:
            import modules.sample
            modules.sample.run()
#    if MODE_EVAL in mode:
#        if MODULE_PRE in modules:
#            preprocess.evaluate()
##        if MODULE_LEXEMES in modules:
##            lexemes.evaluate()
##        if MODULE_DERIV in modules:
##            derivation.evaluate()
#        if MODULE_TRAIN in modules:
#            train.evaluate()
#        if MODULE_ANALYZE in modules:
#            analyze.evaluate()
#    if MODE_EXPORT in mode:
#        if MODULE_PRE in modules:
#            preprocess.export_to_db()
##        if MODULE_LEXEMES in modules:
##            lexemes.export_to_db()
##        if MODULE_DERIV in modules:
##            derivation.export_to_db()
#        if MODULE_TRAIN in modules:
#            train.export_to_db()

if __name__ == '__main__':
    mode, modules = setup()
    main(mode, modules)

