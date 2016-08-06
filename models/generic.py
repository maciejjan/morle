import settings
import pickle

class Model:
    def __init__(self):
        raise Exception('Not implemented!')
    
    def num_rules(self):
        return len(self.rule_features)

    def allows_cost_precomputing(self):
        return False
    
    def cost_of_change(self, edges_to_add, edges_to_remove):
        pass
    
    def apply_change(self, edges_to_add, edges_to_remove):
        pass

    def save_to_file(self, filename):
        with open(settings.WORKING_DIR + filename, 'w+b') as fp:
#            pickle.Pickler(fp, encoding=settings.ENCODING).dump(self)
            pickle.Pickler(fp).dump(self)

    @staticmethod
    def load_from_file(filename):
        with open(settings.WORKING_DIR + filename, 'rb') as fp:
#            pickle.Unpickler(fp, encoding=settings.ENCODING).load()
            return pickle.Unpickler(fp).load()

    def save_rules(self, filename):
        raise NotImplementedError()

    def load_rules(self, filename):
        raise NotImplementedError()
