
# TODO monitor the number of moves from each variant and their acceptance rates!
# TODO refactor
class MCMCGraphSampler:
    def __init__(self, model, lexicon, edges, warmup_iter, sampl_iter):
        self.model = model
        self.lexicon = lexicon
        self.edges = edges
        self.edges_hash = defaultdict(lambda: list())
        self.edges_idx = {}
        for idx, e in enumerate(edges):
            self.edges_idx[e] = idx
            self.edges_hash[(e.source, e.target)].append(e)
#        for idx, e in enumerate(edges):
#            self.edges_hash[(e.source, e.target)] = (idx, e)
        self.len_edges = len(edges)
        self.num = 0        # iteration number
        self.stats = {}
        self.warmup_iter = warmup_iter
        self.sampl_iter = sampl_iter
#         self.tr = tracker.SummaryTracker()
#        self.accept_all = False
    
    def add_stat(self, name, stat):
        if name in self.stats:
            raise Exception('Duplicate statistic name: %s' % name)
        self.stats[name] = stat

    def logl(self):
        return self.model.cost()

    def run_sampling(self):
        logging.getLogger('main').info('Warming up the sampler...')
        pp = progress_printer(self.warmup_iter)
        for i in pp:
            self.next()
        self.reset()
        pp = progress_printer(self.sampl_iter)
        logging.getLogger('main').info('Sampling...')
        for i in pp:
            self.next()
        self.update_stats()

    def next(self):
#         if self.num % 10000 == 0:
#             print(asizeof.asized(self, detail=2).format())
#             for stat_name, stat in self.stats.items():
#                 print(stat_name, asizeof.asized(stat, detail=2).format())
        # increase the number of iterations
        self.num += 1

        # select an edge randomly
        edge_idx = random.randrange(self.len_edges)
        edge = self.edges[edge_idx]

        # try the move determined by the selected edge
        try:
            edges_to_add, edges_to_remove, prop_prob_ratio =\
                self.determine_move_proposal(edge)
            acc_prob = self.compute_acc_prob(\
                edges_to_add, edges_to_remove, prop_prob_ratio)
            if acc_prob >= 1 or acc_prob >= random.random():
                self.accept_move(edges_to_add, edges_to_remove)
            for stat in self.stats.values():
                stat.next_iter(self)
        # if move impossible -- discard this iteration
        except ImpossibleMoveException:
            self.num -= 1

    def determine_move_proposal(self, edge):
        if edge in edge.source.edges:
            return self.propose_deleting_edge(edge)
        elif edge.source.has_ancestor(edge.target):
            return self.propose_flip(edge)
        elif edge.target.parent is not None:
            return self.propose_swapping_parent(edge)
        else:
            return self.propose_adding_edge(edge)

    def propose_adding_edge(self, edge):
        return [edge], [], 1

    def propose_deleting_edge(self, edge):
        return [], [edge], 1

    def propose_flip(self, edge):
        if random.random() < 0.5:
            return self.propose_flip_1(edge)
        else:
            return self.propose_flip_2(edge)

    def propose_flip_1(self, edge):
        edges_to_add, edges_to_remove = [], []
        node_1, node_2, node_3, node_4, node_5 = self.nodes_for_flip(edge)

        if not self.edges_hash[(node_3, node_1)]:
            raise ImpossibleMoveException()

        edge_3_1 = random.choice(self.edges_hash[(node_3, node_1)])
        edge_3_2 = self.find_edge_in_lexicon(node_3, node_2)
        edge_4_1 = self.find_edge_in_lexicon(node_4, node_1)

        if edge_3_2 is not None: edges_to_remove.append(edge_3_2)
        if edge_4_1 is not None:
            edges_to_remove.append(edge_4_1)
        else: raise Exception('!')
        edges_to_add.append(edge_3_1)
        prop_prob_ratio = (1/len(self.edges_hash[(node_3, node_1)])) /\
                          (1/len(self.edges_hash[(node_3, node_2)]))

        return edges_to_add, edges_to_remove, prop_prob_ratio

    def propose_flip_2(self, edge):
        edges_to_add, edges_to_remove = [], []
        node_1, node_2, node_3, node_4, node_5 = self.nodes_for_flip(edge)

        if not self.edges_hash[(node_3, node_5)]:
            raise ImpossibleMoveException()

        edge_2_5 = self.find_edge_in_lexicon(node_2, node_5)
        edge_3_2 = self.find_edge_in_lexicon(node_3, node_2)
        edge_3_5 = random.choice(self.edges_hash[(node_3, node_5)])

        if edge_2_5 is not None:
            edges_to_remove.append(edge_2_5)
        elif node_2 != node_5: raise Exception('!')
        if edge_3_2 is not None: edges_to_remove.append(edge_3_2)
        edges_to_add.append(edge_3_5)
        prop_prob_ratio = (1/len(self.edges_hash[(node_3, node_5)])) /\
                          (1/len(self.edges_hash[(node_3, node_2)]))

        return edges_to_add, edges_to_remove, prop_prob_ratio

    def nodes_for_flip(self, edge):
        node_1, node_2 = edge.source, edge.target
        node_3 = node_2.parent\
                              if node_2.parent is not None\
                              else None
        node_4 = node_1.parent
        node_5 = node_4
        if node_5 != node_2:
            while node_5.parent != node_2: 
                node_5 = node_5.parent
        return node_1, node_2, node_3, node_4, node_5

    def find_edge_in_lexicon(self, source, target):
        edges = [e for e in source.edges if e.target == target] 
        return edges[0] if edges else None

    def propose_swapping_parent(self, edge):
        return [edge], [e for e in edge.target.parent.edges\
                          if e.target == edge.target], 1

    def compute_acc_prob(self, edges_to_add, edges_to_remove, prop_prob_ratio):
        return math.exp(\
                -self.model.cost_of_change(edges_to_add, edges_to_remove)) *\
               prop_prob_ratio

    def accept_move(self, edges_to_add, edges_to_remove):
#            print('Accepted')
        # remove edges and update stats
        for e in edges_to_remove:
            idx = self.edges_idx[e]
            self.lexicon.remove_edge(e)
            self.model.apply_change([], [e])
            for stat in self.stats.values():
                stat.edge_removed(self, idx, e)
        # add edges and update stats
        for e in edges_to_add:
            idx = self.edges_idx[e]
            self.lexicon.add_edge(e)
            self.model.apply_change([e], [])
            for stat in self.stats.values():
                stat.edge_added(self, idx, e)
    
    def reset(self):
        self.num = 0
        for stat in self.stats.values():
            stat.reset(self)

    def update_stats(self):
        for stat in self.stats.values():
            stat.update(self)

    def print_scalar_stats(self):
        stats, stat_names = [], []
        print()
        print()
        print('SIMULATION STATISTICS')
        print()
        spacing = max([len(stat_name)\
                       for stat_name, stat in self.stats.items() 
                           if isinstance(stat, ScalarStatistic)]) + 2
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, ScalarStatistic):
                print((' ' * (spacing-len(stat_name)))+stat_name, ':', stat.value())
        print()
        print()

    def log_scalar_stats(self):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, ScalarStatistic):
                logging.getLogger('main').info('%s = %f' % (stat_name, stat.value()))

    def save_edge_stats(self, filename):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, EdgeStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
        with open_to_write(filename) as fp:
            write_line(fp, ('word_1', 'word_2', 'rule') + tuple(stat_names))
            for i, edge in enumerate(self.edges):
                write_line(fp, 
                           (str(edge.source), str(edge.target), 
                            str(edge.rule)) + tuple([stat.value(i, edge)\
                                                     for stat in stats]))

    def save_rule_stats(self, filename):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, RuleStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
        with open_to_write(filename) as fp:
            write_line(fp, ('rule', 'domsize') + tuple(stat_names))
            for rule in self.model.rule_features:
                write_line(fp, (str(rule), self.model.rule_features[rule][0].trials) +\
                               tuple([stat.value(rule) for stat in stats]))

    def save_wordpair_stats(self, filename):
        stats, stat_names = [], []
        keys = set()
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, WordpairStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
                for (idx_1, idx_2) in stat.values:
                    keys.add((stat.words[idx_1], stat.words[idx_2]))
        with open_to_write(filename) as fp:
            write_line(fp, ('word_1', 'word_2', 'rule') + tuple(stat_names))
            for (word_1, word_2) in sorted(list(keys)):
                write_line(fp, 
                           (word_1, word_2) + tuple([stat.value(word_1, word_2)\
                                                     for stat in stats]))
            
    def summary(self):
        self.print_scalar_stats()
        self.save_edge_stats(shared.filenames['sample-edge-stats'])
        self.save_rule_stats(shared.filenames['sample-rule-stats'])
        self.save_wordpair_stats(shared.filenames['sample-wordpair-stats'])

# TODO constructor arguments should be the same for every type
#      (pass ensured edges through the lexicon parameter?)
# TODO init_lexicon() at creation
class MCMCSemiSupervisedGraphSampler(MCMCGraphSampler):
    def __init__(self, model, lexicon, edges, ensured_conn, warmup_iter, sampl_iter):
        MCMCGraphSampler.__init__(self, model, lexicon, edges, warmup_iter, sampl_iter)
        self.ensured_conn = ensured_conn

    def determine_move_proposal(self, edge):
        edges_to_add, edges_to_remove, prop_prob_ratio =\
            MCMCGraphSampler.determine_move_proposal(self, edge)
        removed_conn = set((e.source, e.target) for e in edges_to_remove) -\
                set((e.source, e.target) for e in edges_to_add)
        if removed_conn & self.ensured_conn:
            raise ImpossibleMoveException()
        else:
            return edges_to_add, edges_to_remove, prop_prob_ratio


class MCMCSupervisedGraphSampler(MCMCGraphSampler):
    def __init__(self, model, lexicon, edges, warmup_iter, sampl_iter):
        logging.getLogger('main').debug('Creating a supervised graph sampler.')
        MCMCGraphSampler.__init__(self, model, lexicon, edges, warmup_iter, sampl_iter)
        self.init_lexicon()

    def init_lexicon(self):
        edges_to_add = []
        for key, edges in self.edges_hash.items():
            edges_to_add.append(random.choice(edges))
        self.accept_move(edges_to_add, [])

    def determine_move_proposal(self, edge):
        if edge in edge.source.edges:
            edge_to_add = random.choice(self.edges_hash[(edge.source, edge.target)])
            if edge_to_add == edge:
                raise ImpossibleMoveException()
            return [edge_to_add], [edge], 1
        else:
            edge_to_remove = self.find_edge_in_lexicon(edge.source, edge.target)
            return [edge], [edge_to_remove], 1

    def run_sampling(self):
        self.reset()
        MCMCGraphSampler.run_sampling(self)


# TODO semi-supervised
class MCMCGraphSamplerFactory:
    def new(*args):
        if shared.config['General'].getboolean('supervised'):
            return MCMCSupervisedGraphSampler(*args)
        else:
            return MCMCGraphSampler(*args)

