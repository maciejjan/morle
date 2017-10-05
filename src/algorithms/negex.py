class NegativeExampleSampler:
    # TODO this is the natural place to store domsizes
    # TODO sample examples for each rule separately
    # TODO sample for each rule as many negative examples
    #      as there are edges with this rule (= potential positive examples)
    # TODO stores also weights of sample items (domsize/sample_size for each rule)
    def sample(self) -> EdgeSet:
        # TODO returns edges (with empty target) and weights
        # TODO automaton: Lex .o. Rules .o. (Lex^c)
        # TODO memorize automata or compute them on the fly?
        raise NotImplementedError()
