from utils.files import open_to_write, read_tsv_file, write_line
import shared


def load_evaluation_data():
    vocab = set(word for (word,) in read_tsv_file(\
                shared.filenames['eval.graph.vocab']))
    # TODO sorting words inside pair: not here but when the files are written!!!
    edges = [(min(word_1, word_2), max(word_1, word_2))\
             for word_1, word_2 in read_tsv_file(
                 shared.filenames['eval.graph'], types=(str, str))]
    edges.sort()
    return vocab, edges

def load_experiment_data():
    reader = read_tsv_file(shared.filenames['sample-wordpair-stats'])
    next(reader)    # skip the header
    edges = [(min(word_1, word_2), max(word_1, word_2), float(prob)) 
             for word_1, word_2, prob in reader if float(prob) > 0.0]
    edges.sort()
    return edges

def evaluate(edges, eval_edges, eval_vocab, prob_bins):
    tp = [0] * len(prob_bins)
    fp = [0] * len(prob_bins)
    fn = [0] * len(prob_bins)

    i_edges = iter(edges)
    i_eval_edges = iter(eval_edges)
    cur_edge = next(i_edges)
    cur_eval_edge = next(i_eval_edges)

    with open_to_write(shared.filenames['eval.report']) as evalfp:
        while True:
            if cur_edge[:2] == cur_eval_edge[:2]:
                write_line(evalfp, cur_edge + (1.0,))
                prob = cur_edge[2]
                for i, pb in enumerate(prob_bins):
                    if prob > pb:
                        tp[i] += 1
                    else:
                        fn[i] += 1
                try:
                    cur_edge = next(i_edges)
                    cur_eval_edge = next(i_eval_edges)
                except StopIteration:
                    break
            elif cur_edge[:2] < cur_eval_edge[:2]:
                if cur_edge[0] in eval_vocab or cur_edge[1] in eval_vocab:
                    write_line(evalfp, cur_edge + (0.0,))
                    prob = cur_edge[2]
                    for i, pb in enumerate(prob_bins):
                        if prob > pb:
                            fp[i] += 1
                try:
                    cur_edge = next(i_edges)
                except StopIteration:
                    break
            else:
                if cur_eval_edge[0] in eval_vocab or cur_eval_edge[1] in eval_vocab:
                    write_line(evalfp, cur_eval_edge + (0.0, 1.0))
                    for i in range(len(prob_bins)):
                        fn[i] += 1
                try:
                    cur_eval_edge = next(i_eval_edges)
                except StopIteration:
                    break

    precision, recall, fscore = [], [], []
    for i in range(len(prob_bins)):
        precision.append(tp[i] / (tp[i] + fp[i]))
        recall.append(tp[i] / (tp[i] + fn[i]))
        fscore.append(2 / (1 / precision[-1] + 1 / recall[-1]))

    return precision, recall, fscore

def print_results(precision, recall, fscore, prob_bins):
    for i in range(len(prob_bins)):
        print('{}\t{:2.2f}\t{:2.2f}\t{:2.2f}\t'.format(
            prob_bins[i], precision[i]*100, recall[i]*100, fscore[i]*100))

def run():
    eval_vocab, eval_edges = load_evaluation_data()
    edges = load_experiment_data()
    prob_bins = [i/100 for i in range(10)] + [i/10 for i in range(1, 10)]
    precision, recall, fscore = \
        evaluate(edges, eval_edges, eval_vocab, prob_bins)
    print_results(precision, recall, fscore, prob_bins)

