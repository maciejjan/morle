import shared

import argparse
# import configparser
import importlib
import logging
import os
import os.path
import re
#import warnings


def process_config():
    '''Load configuration from the working directory. If missing, load
       the default configuration and save it in the working directory.'''

    def load_config(path):
#         shared.config = configparser.ConfigParser()
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

    file_handler = logging.FileHandler(logfile, 'a', encoding=encoding)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(lambda x: re.match('.*[1-9]% done', x.msg) is None)
    logger.addHandler(file_handler)


def setup():
    'Process command-line parameters and update config.'
    ap = argparse.ArgumentParser(prog='morle', \
        description='morle -- MORphemeLEss MORphology LEarner 0.1 alpha')
    # obligatory arguments
#     ap.add_argument('mode', type=str, help='import, export, eval, run')
    ap.add_argument('modules', type=str, help='preprocess, modsel, fit')
    # optional arguments
    ap.add_argument('-d', action='store', dest='workdir', help='working directory')
#     ap.add_argument('-s', action='store_true', dest='supervised', help='supervised learning')
#     ap.add_argument('-t', action='store_true', dest='use_tags', help='use POS-tags')
    ap.add_argument('-q', action='store_true', dest='quiet',
                          help='quiet mode: print less console output')
    ap.add_argument('-v', action='store_true', dest='verbose',
                          help='verbose mode: print more information in the log file')
    ap.add_argument('--version', action='version', version='morle 0.1 alpha')
#    ap.add_argument('-p', '--progress', action='store_true', help='print progress of performed operations')
    args = ap.parse_args()
    if args.workdir is not None:
        shared.options['working_dir'] = os.path.normpath(args.workdir)
    else:
        shared.options['working_dir'] = os.getcwd()
    if not os.path.isdir(shared.options['working_dir']):
        raise RuntimeError('%s: the supplied working directory does not exist!' %\
                           shared.options['working_dir'])
    shared.options['quiet'] = args.quiet
    shared.options['verbose'] = args.verbose
    process_config()
    setup_logger(args.quiet, args.verbose)
    return args.modules.split('+')


def main(modules_to_run):
    for module_name in modules_to_run:
        module = importlib.import_module('modules.' + module_name)
        module.run()


if __name__ == '__main__':
    modules = setup()
    main(modules)

