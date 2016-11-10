from models.generic import Model

class MarginalModel(Model):
    
    def __init__(self, lexicon=None, rules=None, rule_domsizes=None):
        self.model_type = 'marginal'
        Model.__init__(self, lexicon, rules, rule_domsizes)

