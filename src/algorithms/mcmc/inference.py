
class RuleSetProposalDistribution:
    def __init__(self, rule_scores :Dict[Rule, float], 
                 temperature :float) -> None:
        self.rule_prob = {}     # type: Dict[Rule, float]
        for rule, score in rule_scores.items():
#             rule_score = -rule_costs[rule] +\
#                 (rule_contrib[rule] if rule in rule_contrib else 0)
            self.rule_prob[rule] = expit(score * temperature)

    def propose(self) -> Set[Rule]:
        next_ruleset = set()        # type: Set[Rule]
        for rule, prob in self.rule_prob.items():
            if random.random() < prob:
                next_ruleset.add(rule)
        return next_ruleset

    def proposal_logprob(self, ruleset :Set[Rule]) -> float:
        return sum((np.log(prob) if rule in ruleset else np.log(1-prob)) \
                   for rule, prob in self.rule_prob.items())


class MCMCRuleOptimizer:
    def __init__(self, model :Model, full_graph :FullGraph,
                 warmup_iter :int = 0, sampl_iter: int = 0, 
                 alpha :float = 1, beta :float = 0.01) -> None:
        self.iter_num = 0
        self.model = model
        self.full_graph = full_graph
#         self.lexicon = lexicon
#         self.edges = edges
#         self.full_ruleset = set(model.rule_features)
#         self.full_ruleset = self.model.ruleset
#         self.current_ruleset = self.full_ruleset
#         self.full_model = self.model
        self.current_ruleset = set(self.model.rule_features.keys())
        self.rule_domsize = {}      # type: Dict[Rule, int]
#         self.rule_costs = {}        # type: Dict[Rule, float]
        self.warmup_iter = warmup_iter
        self.sampl_iter = sampl_iter
        self.alpha = alpha
        self.beta = beta
        self.update_temperature()
        for rule in self.current_ruleset:
            self.rule_domsize[rule] = \
                self.model.rule_features[rule][0].trials
#             self.rule_costs[rule] = \
#                 self.model.rule_cost(rule, self.rule_domsize[rule])
        self.cost, self.proposal_dist = \
            self.evaluate_proposal(self.current_ruleset)

    def next(self):
        logging.getLogger('main').debug('temperature = %f' % self.temperature)
        next_ruleset = self.proposal_dist.propose()
#        self.print_proposal(next_ruleset)
        cost, next_proposal_dist = self.evaluate_proposal(next_ruleset)
        acc_prob = 1 if cost < self.cost else \
            math.exp((self.cost - cost) * self.temperature) *\
            math.exp(next_proposal_dist.proposal_logprob(self.ruleset) -\
                     self.proposal_dist.proposal_logprob(next_ruleset))
        logging.getLogger('main').debug('acc_prob = %f' % acc_prob)
        if random.random() < acc_prob:
            self.cost = cost
            self.proposal_dist = next_proposal_dist
            self.accept_ruleset(next_ruleset)
            logging.getLogger('main').debug('accepted')
        else:
            logging.getLogger('main').debug('rejected')
        self.iter_num += 1
        self.update_temperature()

    def evaluate_proposal(self, ruleset :Set[Rule]) \
                         -> Tuple[float, RuleSetProposalDistribution]:
#        self.model.reset()
        new_model = MarginalModel()
        new_model.rootdist = self.model.rootdist
        new_model.ruledist = self.model.ruledist
#        new_model.roots_cost = self.model.roots_cost
        for rule in ruleset:
            new_model.add_rule(rule, self.rule_domsize[rule])
#         self.lexicon.reset()
#         new_model.reset()
#         new_model.add_lexicon(self.lexicon)
#        print(new_model.roots_cost, new_model.rules_cost, new_model.edges_cost, new_model.cost())

#         graph_sampler = MCMCGraphSamplerFactory.new(new_model, self.lexicon,\
#             [edge for edge in self.edges if edge.rule in ruleset],\
#             self.warmup_iter, self.sampl_iter)
        graph_sampler = MCMCGraphSamplerFactory.new(
                            new_model, 
                            self.full_graph.restriction_to_ruleset(ruleset),
                            warmup_iter=self.warmup_iter,
                            sampling_iter=self.sampling_iter)
        graph_sampler.add_stat('cost', ExpectedCostStatistic(graph_sampler))
        graph_sampler.add_stat('acc_rate', AcceptanceRateStatistic(graph_sampler))
        graph_sampler.add_stat('contrib', RuleExpectedContributionStatistic(graph_sampler))
        graph_sampler.run_sampling()
        graph_sampler.log_scalar_stats()

        return graph_sampler.stats['cost'].val,\
            RuleSetProposalDistribution(
                graph_sampler.stats['contrib'].values,
                self.rule_costs, self.temperature)

    def accept_ruleset(self, new_ruleset):
        for rule in self.ruleset - new_ruleset:
            self.model.remove_rule(rule)
        for rule in new_ruleset - self.ruleset:
            self.model.add_rule(rule, self.rule_domsize[rule])
        self.current_ruleset = new_ruleset

    def print_proposal(self, new_ruleset):
        for rule in self.ruleset - new_ruleset:
            print('delete: %s' % str(rule))
        for rule in new_ruleset - self.ruleset:
            print('restore: %s' % str(rule))

    def update_temperature(self):
        self.temperature = (self.iter_num + self.alpha) * self.beta

    def save_rules(self, filename):
        self.model.save_rules_to_file(filename)
#         with open_to_write(filename) as outfp:
#             for rule, freq, domsize in read_tsv_file(shared.filenames['rules']):
#                 if Rule.from_string(rule) in self.model.rule_features:
#                     write_line(outfp, (rule, freq, domsize))

    def save_graph(self, filename):
        raise NotImplementedError()
#         with open_to_write(filename) as outfp:
#             for w1, w2, rule in read_tsv_file(shared.filenames['graph']):
#                 if Rule.from_string(rule) in self.model.rule_features:
#                     write_line(outfp, (w1, w2, rule))


#### AUXILIARY FUNCTIONS ###


def load_edges(filename):
    return list(read_tsv_file(filename, (str, str, str)))


# TODO deprecated
# def save_intervals(intervals, filename):
#     with open_to_write(filename) as fp:
#         for rule, ints in intervals.items():
#             write_line(fp, (rule, len(ints), ' '.join([str(i) for i in ints])))


def mcmc_inference(model :Model, full_graph :FullGraph) -> None:
    # initialize the rule sampler
    warmup_iter = shared.config['modsel'].getint('warmup_iterations')
    sampling_iter = shared.config['modsel'].getint('sampling_iterations')
    alpha = shared.config['modsel'].getfloat('annealing_alpha')
    beta = shared.config['modsel'].getfloat('annealing_beta')
    rule_sampler = MCMCAnnealingRuleSampler(
                       model, full_graph, warmup_iter=warmup_iter,
                       sampling_iter=sampling_iter, alpha=alpha, beta=beta)
    # main loop -- perfom the inference
    iter_num = 0
    while iter_num < shared.config['modsel'].getint('iterations'):
        iter_num += 1
        logging.getLogger('main').info('Iteration %d' % iter_num)
        logging.getLogger('main').info(\
            'num_rules = %d' % rule_sampler.model.num_rules())
        logging.getLogger('main').info('cost = %f' % rule_sampler.cost)
#         rule_sampler.next()
        rule_sampler.save_rules(shared.filenames['rules-modsel'])
        rule_sampler.save_graph(shared.filenames['graph-modsel'])

