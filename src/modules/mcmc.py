### MODULE FUNCTIONS ###

def run():
	ruleset = RuleSet.load_from_file(settings.FILES['model.rules'] + '.0')
	lexicon = Lexicon.init_from_file(settings.FILES['training.wordlist'],\
		ruleset=ruleset)
	edges = load_edges(settings.FILES['surface.graph'])
	optimize_rules(lexicon, edges, settings.FILES['model.rules'])

def run_old():
	ruleset = RuleSet.load_from_file(settings.FILES['model.rules'] + '.0')
	lexicon = Lexicon.init_from_file(settings.FILES['training.wordlist'],\
		ruleset=ruleset)
	edges = load_edges(settings.FILES['surface.graph'])
	sampler = MCMCGraphSampler(lexicon, edges)
	sampler.add_stat('exp_rule_freq', RuleFrequencyStatistic(sampler))
	sampler.add_stat('graphs_without', RuleGraphsWithoutStatistic(sampler))
	sampler.add_stat('int_without', RuleIntervalsWithoutStatistic(sampler))
	sampler.add_stat('exp_edge_freq', EdgeFrequencyStatistic(sampler))
	sampler.add_stat('change_count', RuleChangeCountStatistic(sampler))
	sampler.add_stat('time', TimeStatistic(sampler))
	sampler.add_stat('exp_logl', ExpectedLogLikelihoodStatistic(sampler))
	sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
	print('Warming up the chain...')
	pp = progress_printer(NUM_WARMUP_ITERATIONS)
	while sampler.num < NUM_WARMUP_ITERATIONS:
		num = sampler.num
		sampler.next()
		if sampler.num > num: next(pp)
	sampler.reset()
	print('Sampling graphs...')
	sample = np.zeros(NUM_ITERATIONS)
	with open_to_write('logl.txt') as fp:
		pp = progress_printer(NUM_ITERATIONS)
		while sampler.num < NUM_ITERATIONS:
			num = sampler.num
			sampler.next()
			if sampler.num > num:
				next(pp)
				sample[sampler.num-1] = sampler.logl
			if sampler.num % 1000 == 0:
				write_line(fp, (sampler.logl, sampler.stats['exp_logl'].value()))
	sampler.update_stats()
	print_scalar_stats(sampler)
	save_edges(sampler, 'graph-mcmc.txt')
	save_rules(sampler, 'rules-mcmc.txt')
	save_intervals(sampler.stats['int_without'].intervals, 'intervals.txt')
	print('Sampling rules...')
	rule_sampler = MCMCRuleSetSampler(sample, sampler.stats['int_without'].intervals, sampler.lexicon.ruleset, ANNEALING_TEMPERATURE)
	with open_to_write('sampling-rules.log') as logfp:
		pp = progress_printer(NUM_RULE_ITERATIONS)
		while rule_sampler.num < NUM_RULE_ITERATIONS:
			next(pp)
			rule_sampler.next(logfp)
	rule_sampler.save_best_ruleset(settings.FILES['model.rules'])

def evaluate():
	pass

