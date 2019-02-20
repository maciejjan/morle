from utils.files import open_to_write, read_tsv_file, write_line
import shared


'''Perform naive model selection and fitting based on sampling results:
   * remove rules with expected contribution <= 0
   * set productivity to exp. frequency / domsize'''

def run() -> None:
    reader = read_tsv_file(shared.filenames['sample-rule-stats'])
    col_labels = next(reader)
    rule_col = col_labels.index('rule')
    domsize_col = col_labels.index('domsize')
    contrib_col = col_labels.index('contrib')
    freq_col = col_labels.index('freq')
    with open_to_write(shared.filenames['rules-fit']) as fp:
        for row in reader:
            rule = row[rule_col]
            domsize = int(row[domsize_col])
            contrib = float(row[contrib_col])
            freq = float(row[freq_col])
            if contrib > 0:
                prod = freq / domsize
                write_line(fp, (rule, domsize, prod))

