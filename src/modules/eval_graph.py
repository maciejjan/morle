# TODO
# - take the sampled graph
# - sum the probability for each word pair over all rule labels!
#   (they are mutually exclusive, so we can sum)
# - sort the resulting edges
# - compare to gold standard edges
#   - produce a list: word_1 word_2 prob gs_prob (0 or 1)
# - display a table of results with various thresholds

def choose_edges_for_evaluation(graph_file, eval_vocab_file, eval_filename):
    eval_vocab = set()
    for (word,) in read_tsv_file(eval_vocab_file):
        eval_vocab.add(word)
    graph_reader = read_tsv_file(graph_file, types=(str, str, float))
    next(graph_reader)      # skip the header
    with open_to_write(eval_filename) as eval_fp:
        for (word_1, word_2, prob) in graph_reader:
            if word_1 in eval_vocab:
                write_line(evalfp, (word_1, word_2, prob))
    sort_file(eval_fp, key=1)
    # TODO

def evaluate_edges():
    raise NotImplementedError()

def compute_evaluation_measures():
    raise NotImplementedError()

def run():
    # format: word_1 word_2 prob gs_prob
    raise NotImplementedError()
