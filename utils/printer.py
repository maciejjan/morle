import shared
import logging
import sys

def write_to_stdout(msg):
    if not shared.options['quiet']:
        sys.stdout.write(msg)
        sys.stdout.flush()

def progress_printer(max_count):
    count = 0
    progress = 0
    write_to_stdout('--> 0 %')
    while count < max_count:
        count += 1
        new_progress = 100 * count // max_count
        if new_progress > progress:
            progress = new_progress
            write_to_stdout('\r--> ' + str(progress) + ' %')
            logging.getLogger('main').info(str(progress) + '% done')
        if new_progress == 100:
            write_to_stdout('\n')
        yield

