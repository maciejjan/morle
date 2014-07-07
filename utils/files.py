import settings
import codecs
import os
import os.path
import utils.printer

# TODO types as parameter

def read_tsv_file(filename, types=None, print_progress=False, print_msg=None):
	pp = None
	if print_progress:
		print print_msg
		pp = utils.printer.progress_printer(get_file_size(filename))
	with codecs.open(settings.WORKING_DIR+filename, 'r', settings.ENCODING) as fp:
		for line in fp:
			row = line.rstrip().split('\t')
			if isinstance(types, tuple):
				row_converted = []
				for i in range(min(len(row), len(types))):
#					row[i] = types[i](row[i])
					row_converted.append(types[i](row[i]))
				row = row_converted
			yield tuple(row)
			if print_progress:
				pp.next()

def read_tsv_file_by_key(filename, key_col=1, types=None,\
		print_progress=False, print_msg=None):
	current_key, entries = None, []
	for row in read_tsv_file(filename, types, print_progress, print_msg):
		key, entry = None, None
		if isinstance(key_col, tuple):
			key = tuple([row[i-1] for i in key_col])
			entry = tuple([row[i] for i in range(0, len(row)) if not i+1 in key_col])
		elif isinstance(key_col, int):
			key = row[key_col-1]
			entry = tuple([row[i] for i in range(0, len(row)) if i+1 != key_col])
		if key != current_key:
			if entries:
				yield current_key, entries
			current_key, entries = key, []
		entries.append(entry)
	if entries:
		yield current_key, entries

def open_to_write(filename):
	return codecs.open(settings.WORKING_DIR + filename, 'w+', settings.ENCODING)

# if count_bytes set to true, returns the number of bytes written
def write_line(fp, line, count_bytes=False):
	line = u'\t'.join([unicode(s) for s in line]) + u'\n'
	fp.write(line)
	if count_bytes:
		return len(line.encode(settings.ENCODING))

FILE_SIZES = {}

def count_lines(filename):
	num = 0
	with codecs.open(settings.WORKING_DIR + filename, 'r', settings.ENCODING) as fp:
		for line in fp:
			num += 1
	return num

def get_file_size(filename):
	global FILE_SIZES
	if FILE_SIZES.has_key(filename):
		return FILE_SIZES[filename]
	else:
		read_index_file()
		if FILE_SIZES.has_key(filename):
			return FILE_SIZES[filename]
		else:
			size = count_lines(filename)
			set_file_size(filename, size)
			return size

def read_index_file():
	global FILE_SIZES
	if os.path.isfile(settings.WORKING_DIR + settings.FILES['index']):
		for filename, size in read_tsv_file(settings.FILES['index']):
			if not FILE_SIZES.has_key(filename):
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
	with open_to_write(settings.FILES['index']) as fp:
		for filename, size in FILE_SIZES.iteritems():
			write_line(fp, (filename, size))

def file_exists(filename):
	return os.path.isfile(filename)

def rename_file(old, new):
	os.rename(settings.WORKING_DIR + old, settings.WORKING_DIR + new)

def remove_file(filename):
	os.remove(settings.WORKING_DIR + filename)

# sort file using the unix command
def sort_file(infile, outfile=None, key=None, reverse=False, numeric=False, stable=False, unique=False):
	sort_call = ['sort', settings.WORKING_DIR + infile]
	if key:
		if isinstance(key, tuple) and len(key) == 2:
			sort_call.append('-k%d,%d' % key)
			sort_call.append('-t \'	\'')
		elif isinstance(key, int):
			sort_call.append('-k%d,%d' % (key, key))
			sort_call.append('-t \'	\'')
	if reverse:
		sort_call.append('-r')
	if numeric:
		sort_call.append('-n')
	if stable:
		sort_call.append('-s')
	if unique:
		sort_call.append('-u')
	sort_call.append('-o')
	if outfile:
		sort_call.append(settings.WORKING_DIR + outfile)
	else:
		sort_call.append(settings.WORKING_DIR + infile + '.sorted')
	print ' '.join(sort_call)
	os.system(' '.join(sort_call))
	if outfile is None:
		remove_file(infile)
		rename_file(infile + '.sorted', infile)

