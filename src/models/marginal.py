from models.generic import Model

class MarginalModel(Model):
    
    def __init__(self):
        self.model_type = 'marginal'
        Model.__init__(self)

