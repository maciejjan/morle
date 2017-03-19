import shared

import csv
import logging
import numpy as np
import os
import os.path
import tqdm
from typing import Any, Dict, Iterable, List, Tuple, Union
from typing.io import TextIO


def full_path(filename :str) -> str:
    return os.path.join(shared.options['working_dir'], filename)


def read_tsv_file(filename :str,
                  types :Iterable = None,
                  show_progressbar=False) -> Iterable[List[Any]]:

    progressbar = None
    if show_progressbar:
        progressbar = tqdm.tqdm(total=get_file_size(filename))
    encoding = shared.config['General'].get('encoding')
    converter = (lambda row: [t(row[i]) for i, t in enumerate(types) \
                                        if callable(t)]) \
                if types is not None \
                else lambda row: row

    with open(full_path(filename), 'r', encoding=encoding, newline='') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for row in reader:
            yield converter(row)
            if show_progressbar:
                progressbar.update()
    if show_progressbar:
        progressbar.close()


def read_tsv_file_by_key(filename :str,
                         key :Union[int, Iterable[int]] = 1,
                         types :Iterable = None,
                         show_progressbar :bool = False) \
    -> Iterable[Tuple[Any, List[List[Any]]]]:

    key_and_entry_maker = None
    if isinstance(key, int):
        key_and_entry_maker = lambda x: (x[key-1], x[:key-1]+x[key:])
    elif hasattr(key, '__contains__') and hasattr(key, '__iter__'):
        key_and_entry_maker = \
            lambda x: ([x[i-1] for i in key], \
                       [x[i] for i in range(len(x)) if i+1 not in key])
    else:
        raise RuntimeError('Invalid key: %s' % str(key))

    current_key, entries = None, []
    for row in read_tsv_file(filename, types, show_progressbar):
        key_, entry = key_and_entry_maker(row)
        if key_ != current_key:
            if entries:
                yield current_key, entries
            current_key, entries = key_, []
        entries.append(entry)
    if entries:
        yield current_key, entries


def open_to_write(filename :str, mode :str = 'w+') -> TextIO:
    encoding = shared.config['General'].get('encoding')
    return open(full_path(filename), mode, encoding=encoding)


# if count_bytes set to true, returns the number of bytes written
def write_line(fp, line, count_bytes=False):
    line = u'\t'.join([str(s) for s in line]) + u'\n'
    fp.write(line)
    if count_bytes:
        return len(bytes(line))


def write_tsv_file(filename :str, row_iter :Iterable) -> None:
    with open(full_path(filename), 'w+', newline='') as fp:
        writer = csv.writer(fp, delimiter='\t')
        for row in row_iter:
            writer.writerow(row)


FILE_SIZES = {} # type: Dict[str, int]


def count_lines(filename :str) -> int:
    num = 0
    encoding = shared.config['General'].get('encoding')
    with open(full_path(filename), 'r', encoding=encoding) as fp:
        for line in fp:
            num += 1
    return num


def get_file_size(filename :str) -> int:
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


def read_index_file() -> None:
    global FILE_SIZES
    if os.path.isfile(full_path(shared.filenames['index'])):
        for filename, size in read_tsv_file(shared.filenames['index']):
            if filename not in FILE_SIZES:
                FILE_SIZES[filename] = int(size)


def set_file_size(filename :str, size :int) -> None:
    global FILE_SIZES
    FILE_SIZES[filename] = size
    update_index_file()


def update_file_size(filename :str) -> None:
    set_file_size(filename, count_lines(filename))


def update_index_file() -> None:
    global FILE_SIZES
    read_index_file()
    with open_to_write(shared.filenames['index']) as fp:
        for filename, size in FILE_SIZES.items():
            write_line(fp, (filename, size))


def file_exists(filename :str) -> bool:
    return os.path.isfile(full_path(filename))


def rename_file(old :str, new :str) -> None:
    os.rename(full_path(old), full_path(new))


def remove_file(filename :str) -> None:
    os.remove(full_path(filename))


def remove_file_if_exists(filename :str) -> None:
    path = full_path(filename)
    if os.path.isfile(path):
        os.remove(path)


# sort file using the unix command
def sort_file(infile, outfile=None, key=None, reverse=False, numeric=False, 
              stable=False, unique=False, parallel=None):
    sort_call = ['sort']
#     if isinstance(infiles, str):
    sort_call.append(full_path(infile))
#     elif isinstance(infiles, list):
#         sort_call.extend([full_path(infile) for infile in infiles])
#     else:
#         raise RuntimeError('sort: wrong input type!')
    sort_call += ['-T', shared.options['working_dir']]
    if key:
        if isinstance(key, tuple) and len(key) == 2:
            sort_call.append('-k%d,%d' % key)
            sort_call.append('-t \'	\'')
        elif isinstance(key, int):
            sort_call.append('-k%d,%d' % (key, key))
            sort_call.append('-t \'	\'')
        else:
            raise RuntimeError('Wrong key type.')
    if reverse:
        sort_call.append('-r')
    if numeric:
        sort_call.append('-g')
        sort_call.insert(0, 'LC_NUMERIC=us_EN.UTF-8')
    if stable:
        sort_call.append('-s')
    if unique:
        sort_call.append('-u')
    if parallel is not None and isinstance(parallel, int):
        sort_call.append('--parallel={}'.format(parallel))
    sort_call.append('-o')
    if outfile:
        sort_call.append(full_path(outfile))
    else:
#         if isinstance(infiles, str):
        sort_call.append(full_path(infile) + '.sorted')
#         elif isinstance(infiles, list):
#             sort_call.append(full_path(infiles[0]) + '.sorted')
#         else:
#             raise RuntimeError('sort: wrong input type!')
    logging.getLogger('main').debug(' '.join(sort_call))
    os.system(' '.join(sort_call))
    if outfile is None:
#         if isinstance(infiles, str):
        remove_file(infiles)
        rename_file(infiles + '.sorted', infiles)
#         elif isinstance(infiles, list):
#             for infile in infiles:
#                 remove_file(infile)
#             rename_file(infiles[0] + '.sorted', infiles[0])
#         else:
#             raise RuntimeError('sort: wrong input type!')


def aggregate_file(infile, outfile=None, key=1):
    if outfile is None:
        outfile = infile + '.agg'
    with open_to_write(outfile) as fp:
        for key_, rows in read_tsv_file_by_key(infile, key):
            write_line(fp, (key_, len(rows)))

