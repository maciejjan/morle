import sys

def progress_printer(max_count):
	count = 0
	progress = 0
	sys.stdout.write('--> 0 %')
	sys.stdout.flush()
	while count < max_count:
		count += 1
		new_progress = 100 * count / max_count
		if new_progress > progress:
			progress = new_progress
			sys.stdout.write('\r--> ' + str(progress) + ' %')
			sys.stdout.flush()
		if new_progress == 100:
			print ''
		yield

