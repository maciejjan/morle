import shared
import utils.printer
import logging
import numpy as np
import os
import os.path

def full_path(filename):
    return os.path.join(shared.options['working_dir'], filename)

def read_tsv_file(filename, types=None, print_progress=False, print_msg=None):
    pp = None
    if print_progress:
        if print_msg:
            logging.getLogger('main').info(print_msg)
        pp = utils.printer.progress_printer(get_file_size(filename))
    encoding = shared.config['General'].get('encoding')
    with open(full_path(filename), 'r', encoding=encoding) as fp:
        for line in fp:
            row = line.rstrip().split('\t')
            if isinstance(types, tuple):
                row_converted = []
                i = 0
                for t in types:
                    if callable(t):
                        row_converted.append(t(row[i]))    # t is a conversion function
                        i += 1
                        if i >= len(row): break
                    else:
                        row_converted.append(t)            # t is a value
                row = row_converted
            yield tuple(row)
            if print_progress:
                next(pp)

def read_tsv_file_by_key(filename, key=1, types=None,\
        print_progress=False, print_msg=None):
    current_key, entries = None, []
    for row in read_tsv_file(filename, types, print_progress, print_msg):
        key_, entry = None, None
        if isinstance(key, tuple):
            key_ = tuple([row[i-1] for i in key])
            entry = tuple([row[i] for i in range(0, len(row)) if not i+1 in key])
        elif isinstance(key, int):
            key_ = row[key-1]
            entry = tuple([row[i] for i in range(0, len(row)) if i+1 != key])
        if key_ != current_key:
            if entries:
                yield current_key, entries
            current_key, entries = key_, []
        entries.append(entry)
    if entries:
        yield current_key, entries

def open_to_write(filename, mode='w+'):
    encoding = shared.config['General'].get('encoding')
    return open(full_path(filename), mode, encoding=encoding)

# if count_bytes set to true, returns the number of bytes written
def write_line(fp, line, count_bytes=False):
    line = u'\t'.join([str(s) for s in line]) + u'\n'
    fp.write(line)
    if count_bytes:
        return len(bytes(line))

FILE_SIZES = {}

def count_lines(filename):
    num = 0
    encoding = shared.config['General'].get('encoding')
    with open(full_path(filename), 'r', encoding=encoding) as fp:
        for line in fp:
            num += 1
    return num

def get_file_size(filename):
    global FILE_SIZES
    if filename in FILE_SIZES:
        return FILE_SIZES[filename]
    else:
        read_index_file()
        if filename in FILE_SIZES:
            return FILE_SIZES[filename]
        else:
            size = count_lines(filename)
            set_file_size(filename, size)
            return size

def read_index_file():
    global FILE_SIZES
    if os.path.isfile(full_path(shared.filenames['index'])):
        for filename, size in read_tsv_file(shared.filenames['index']):
            if filename not in FILE_SIZES:
                FILE_SIZES[filename] = int(size)

def set_file_size(filename, size):
    global FILE_SIZES
    FILE_SIZES[filename] = size
    update_index_file()

def update_file_size(filename):
    set_file_size(filename, count_lines(filename))

def update_index_file():
    global FILE_SIZES
    read_index_file()
    with open_to_write(shared.filenames['index']) as fp:
        for filename, size in FILE_SIZES.items():
            write_line(fp, (filename, size))

def file_exists(filename):
    return os.path.isfile(full_path(filename))

def rename_file(old, new):
    os.rename(full_path(old), full_path(new))

def remove_file(filename):
    os.remove(full_path(filename))

def remove_file_if_exists(filename):
    path = full_path(filename)
    if os.path.isfile(path):
        os.remove(path)

# sort file using the unix command
def sort_file(infile, outfile=None, key=None, reverse=False, numeric=False, stable=False, unique=False):
    sort_call = ['sort', full_path(infile)]
    if key:
        if isinstance(key, tuple) and len(key) == 2:
            sort_call.append('-k%d,%d' % key)
            sort_call.append('-t \'    \'')
        elif isinstance(key, int):
            sort_call.append('-k%d,%d' % (key, key))
            sort_call.append('-t \'	\'')
    if reverse:
        sort_call.append('-r')
    if numeric:
        sort_call.append('-g')
        sort_call.insert(0, 'LC_NUMERIC=us_EN.UTF-8')
    if stable:
        sort_call.append('-s')
    if unique:
        sort_call.append('-u')
    sort_call.append('-o')
    if outfile:
        sort_call.append(full_path(outfile))
    else:
        sort_call.append(full_path(infile) + '.sorted')
    logging.getLogger('main').debug(' '.join(sort_call))
    os.system(' '.join(sort_call))
    if outfile is None:
        remove_file(infile)
        rename_file(infile + '.sorted', infile)

def aggregate_file(infile, outfile=None, key=1):
    if outfile is None:
        outfile = infile + '.agg'
    with open_to_write(outfile) as fp:
        for key_, rows in read_tsv_file_by_key(infile, key):
            write_line(fp, (key_, len(rows)))

