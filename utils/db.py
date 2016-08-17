from utils.files import *
from utils.printer import *
import shared
from warnings import filterwarnings
#import MySQLdb

#filterwarnings('ignore', category=MySQLdb.Warning)

### FILE CONVERSION FUNCTIONS ###

def insert_id(input_file, output_file, key=1):
	# insert ID (line number) as first column to a file
	# return a dictionary key => id
	ids = {}
	pp = progress_printer(get_file_size(input_file))
	with open_to_write(output_file) as fp:
		for i, row in enumerate(load_tsv_file(input_file), 1):
			ids[row[key-1]] = i
			write_line(fp, tuple([i] + list(row)))
			pp.next()
	return ids

def replace_values_with_ids(input_file, output_file, ids):
	# replace certain columns with IDs 
	# ids - a tuple of dictionaries mapping each column on an ID
	pp = progress_printer(get_file_size(input_file))
	with open_to_write(output_file) as fp:
		for row in load_tsv_file(input_file):
			new_row = []
			for i, value in enumerate(row):
				new_row.append(ids[i][value] if ids[i] is not None else value)
			write_line(fp, tuple(new_row))
			pp.next()

def decode_unicode(row):
	row_conv = []
	for x in row:
		row_conv.append(x.decode('utf-8') if isinstance(x, str) else x)
	return tuple(row_conv)

### MYSQL CONNECTION FUNCTIONS ###

CONNECTION = None
CURSOR = None

def connect():
	global CONNECTION, CURSOR
	if settings.DB_HOST is None:
		raise Exception("No database host supplied.")
	if settings.DB_NAME is None:
		raise Exception("No database name supplied.")
	if settings.DB_USER is None:
		raise Exception("No database username supplied.")
	if settings.DB_PASS is None:
		raise Exception("No database password supplied.")
	raise Exception('Not implemented yet!')
#	CONNECTION = MySQLdb.connect(host=settings.DB_HOST, db=settings.DB_NAME,
#		user=settings.DB_USER, passwd=settings.DB_PASS, use_unicode=True,\
#		charset='utf8', local_infile=1)
#	CURSOR = CONNECTION.cursor()

def close_connection():
	CURSOR.close()
	CONNECTION.close()

def query(q):
	CURSOR.execute(q)

def query_fetch_results(q):
	CURSOR.execute(q)
	row = CURSOR.fetchone()
	while row is not None:
		yield decode_unicode(row)
		row = CURSOR.fetchone()

def query_fetch_all_results(q):
	CURSOR.execute(q)
	return [decode_unicode(r) for r in CURSOR.fetchall()]

### MYSQL TABLES INPUT/OUTPUT FUNCTIONS ###

def push_table(table, filename):
	tbl_name, tbl_create = table
	query('DROP TABLE IF EXISTS `%s`;' % tbl_name)
	query(tbl_create)
	query('ALTER TABLE %s DISABLE KEYS;' % tbl_name)
	query('LOAD DATA LOCAL INFILE \'%s\' INTO TABLE `%s`;' %\
		(settings.WORKING_DIR + filename, tbl_name))
	query('ALTER TABLE %s ENABLE KEYS;' % tbl_name)

def pull_table(table, fields, filename):
	tbl_name = table[0]
	with open_to_write(filename) as fp:
		for row in query_fetch_results('SELECT %s FROM `%s`;' %\
			(', '.join(['`'+f+'`' for f in fields]), tbl_name)):
			write_line(fp, row)

