#models.py - assign probabilities/log-probabilities to objects
#-	hyperparameters
#	(root distributions etc.)
#-	edge_gain
#-	rule_cost (prior)
#-	word_cost (prior)
#-	cost of a rule with all its edges? (total_rule_cost)
#-	rule_split_gain  (~ improvement_fun from "optrules")
#-	fit parameters to the lexicon (roots distrib.) and rule set
#-	etc.

class Model:
	def edge_gain(self, ruleset, edge):
		# TODO the gain in LogL by drawing a new edge
		raise Exception('Not implemented!')

#-	compute edge costs in advance

#Types of models:
#-	PointModel (parameters set to points)
#-	MarginalModel (parameters integrated out)

# rule data in each of the models:
# - PointModel:
#   - productivity, domsize
#   - if freq. used: freq. multiplier mean and sd
#   - if vec. used: vec. mean and sd
# - MarginalModel:
#   - (exp.) count, domsize
#   - if freq. used: freq. multiplier hyperparameters (mu, kappa, alpha, beta)
#   - if vec. used: vec. hyperparameters (mu, kappa, alpha, beta)

# edge_cost requires rule data as input
