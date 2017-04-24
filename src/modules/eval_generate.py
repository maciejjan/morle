from utils.files import open_to_write, read_tsv_file, write_line
import shared


def run() -> None:
    bins = list(map(int, 
                    shared.config['eval_generate'].get('bins').split(',')))
    known_words = set(word for (word,) in \
                             read_tsv_file(shared.filenames['eval.wordlist']))
    tp, fp = [0] * len(bins), [0] * len(bins)
    with open_to_write(shared.filenames['eval.wordgen']) as outfp:
        for i, (word, base, weight) in \
                enumerate(read_tsv_file(shared.filenames['wordgen'])):
            current_tp = 1 if word in known_words else 0
            current_fp = 1 - current_tp
            for j, length in enumerate(bins):
                if i < length:
                    tp[j] += current_tp
                    fp[j] += current_fp
            write_line(outfp, (word, current_tp, base, weight))
    for j, length in enumerate(bins):
        print(length, '{:.2%}'.format(tp[j]/(tp[j]+fp[j])))

