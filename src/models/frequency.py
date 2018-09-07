from models.generic import Model, ModelFactory


class RootFrequencyModel(Model):
    pass

class ZipfRootFrequencyModel(RootFrequencyModel):
    pass

class RootFrequencyModelFactory(ModelFactory):
    pass

class EdgeFrequencyModel(Model):
    pass

class LogNormalEdgeFrequencyModel(EdgeFrequencyModel):
    pass

class EdgeFrequencyModelFactory(ModelFactory):
    pass
