import algorithms.em
#import algorithms.optrules
from datastruct.lexicon import *
from datastruct.rules import *
from models.point import PointModel, fit_model_to_graph
from utils.files import *
#from utils.printer import *
import settings

def save_analyses(lexicon, filename):
    def analysis(node):
        if node.parent is None:
            return [node]
        else:
            return [node] + analysis(node.parent)
    with open_to_write(filename) as fp:
        for node in sorted(lexicon.values(), key=str):
            write_line(fp, (' <- '.join(map(str, analysis(node))),))

def save_rules(model, filename):
    with open_to_write(filename) as fp:
        for rule, features in model.rule_features.items():
            write_line(fp, (str(rule), features[0].prob, features[0].trials))


def train_unsupervised():
    lexicon = Lexicon.init_from_file(settings.FILES['training.wordlist'])
    model_filename, model = settings.FILES['model'] + '.0', None
    if file_exists(model_filename):
        model = PointModel.load_from_file(model_filename)
    else:
        print('Fitting the initial model...')
        model = fit_model_to_graph(lexicon, settings.FILES['surface.graph'])
        model.save_to_file(model_filename)
    print('Loading edges...')
    edges = [LexiconEdge(lexicon[w1], lexicon[w2], Rule.from_string(r))\
        for w1, w2, r in read_tsv_file(settings.FILES['surface.graph'])]

    # compute node and edge costs
    for node in lexicon.iter_nodes():
        node.cost = model.node_cost(node)
    for edge in edges:
        if model.has_rule(edge.rule):
            edge.cost = model.edge_cost(edge)
    lexicon.recompute_cost(model)

#    with open_to_write('vertices.txt') as fp:
#        for n in lexicon.iter_nodes():
#            write_line(fp, tuple(map(str, (n, n.cost))))
    print('Saving edges...')
    with open_to_write('edges.txt') as fp:
        for e in edges:
            write_line(fp, tuple(map(str, (e.source, e.target, e.rule, e.cost, e.target.cost, e.target.cost-e.cost))))

    if settings.TRAINING_ALGORITHM == 'hardem':
        algorithms.em.hardem(lexicon, model, edges)

    elif settings.TRAINING_ALGORITHM == 'softem':
        algorithms.em.softem(lexicon, model, edges)

    # TODO misleading! 'mcmc' algorithm not in 'mcmc' module
    elif settings.TRAINING_ALGORITHM == 'mcmc':
        algorithms.mcmc.mcmc_inference(lexicon, model, edges)

    print('Saving model...')
    model.save_to_file(settings.FILES['model'])
    save_analyses(lexicon, settings.FILES['analyses'])
    save_rules(model, settings.FILES['model.rules'])

def train_hardem():
    lexicon = Lexicon.init_from_file(settings.FILES['training.wordlist'])
    model = Model.load_from_file(settings.FILES['model'] + '.0')
#    model = fit_model_to_graph(lexicon, settings.FILES['surface.graph'])
#    model.save_to_file(settings.FILES['model'] + '.0')
    print(model.num_rules())
    lexicon.recompute_costs(model)
    i = 0
    old_logl = lexicon.logl
    old_num_rules = model.num_rules()
    possible_edges = [LexiconEdge(lexicon[w1], lexicon[w2], model.rules[r])\
        for w1, w2, r in read_tsv_file(settings.FILES['surface.graph'])]

    with open_to_write('vertices.txt') as fp:
        for n in lexicon.iter_nodes():
            write_line(fp, tuple(map(str, (n, n.cost))))
    with open_to_write('edges.txt') as fp:
        for e in possible_edges:
            if str(e.rule) in model.rules:
                e.weight = model.edge_cost(e) -    model.word_cost(e.target)
            write_line(fp, tuple(map(str, (e.source, e.target, e.rule, model.edge_cost(e), model.word_cost(e.target), e.weight))))

    while True:
        i += 1
        print('\n===   Iteration %d   ===\n' % i)
        print('number of rules:', old_num_rules)
        print('LogL = ', str(lexicon.logl))
#        logl = expectation_maximization(lexicon)
        algorithms.em.hardem_iteration(lexicon, model, possible_edges)
        num_rules = model.num_rules()
        if num_rules == old_num_rules and lexicon.logl - old_logl < 1.0:
            break
        else:
#            lexicon.ruleset.save_to_file(settings.FILES['model.rules'])
            lexicon.save_to_file(settings.FILES['model.lexicon'])
            model.save_to_file(settings.FILES['model'])
            old_logl = lexicon.logl
            old_num_rules = model.num_rules()

    save_analyses(lexicon, settings.FILES['analyses'])

def train_softem():
    lexicon = Lexicon.init_from_file(settings.FILES['training.wordlist'])
    model = Model.load_from_file(settings.FILES['model'] + '.0')
    edges = [LexiconEdge(lexicon[w1], lexicon[w2], model.rules[r])\
        for w1, w2, r in read_tsv_file(DATA_PATH + 'graph.txt')]
    
    lexicon.reset()
    lexicon.recompute_costs(model)
    for edge in edges:
        edge.weight = model.edge_cost(edge)
    
    sampler = algorithms.mcmc.MCMCGraphSampler(model, lexicon, edges)
    sampler.add_stat('exp_edge_freq',\
        algorithms.mcmc.EdgeFrequencyStatistic(sampler))

    old_logl = lexicon.logl
    old_num_rules = model.num_rules()
    while True:
        i += 1
        print('\n===   Iteration %d   ===\n' % i)
        print('number of rules:', old_num_rules)
#        logl = expectation_maximization(lexicon)
        algorithms.em.softem_iteration(lexicon, model, sampler)
        num_rules = model.num_rules()
        if num_rules == old_num_rules and lexicon.logl - old_logl < 1.0:
            break
        else:
#            lexicon.ruleset.save_to_file(settings.FILES['model.rules'])
            lexicon.save_to_file(settings.FILES['model.lexicon'])
            model.save_to_file(settings.FILES['model'])
            old_logl = lexicon.logl
            old_num_rules = model.num_rules()

def train_supervised():
    lexicon = algorithms.mdl.load_training_file(settings.FILES['training.lexicon'])
#    unigrams.save_to_file(settings.FILES['model.ngrams'])
    # compute rules domain size -- TODO
    trh = TrigramHash()
    for word in lexicon.keys():
        trh.add(word)
    print('Calculating rule domain sizes...')
    pp = progress_printer(len(lexicon.ruleset))
    for r in lexicon.ruleset.values():
        r.domsize = algorithms.optrules.rule_domsize(r.rule, trh)
        next(pp)
    algorithms.mdl.optimize_rule_params(lexicon)
    lexicon.save_model(settings.FILES['model.rules'], settings.FILES['model.lexicon'])
    # optimize rules
    algorithms.optrules.optimize_rules_in_lexicon(\
        settings.FILES['model.lexicon'],
        settings.FILES['model.lexicon'] + '.opt',
        settings.FILES['model.rules'])
    rename_file(settings.FILES['model.lexicon'] + '.opt',\
        settings.FILES['model.lexicon'])
    ruleset = RuleSet.load_from_file(settings.FILES['model.rules'])
    lexicon = Lexicon.load_from_file(settings.FILES['model.lexicon'], ruleset)
    algorithms.mdl.optimize_rule_params(lexicon)
    lexicon.save_model(settings.FILES['model.rules'], settings.FILES['model.lexicon'])

def run():
    if settings.SUPERVISED:
        train_supervised()
    else:
        train_unsupervised()

def evaluate():
    pass

