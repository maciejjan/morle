import re

# shared
# .config -- saved in a configuration file in the working dir
# .options -- command-line options
# .filenames
# .patterns

# configuration and working directory -- set at runtime by the 'main' module

config = None

options = {\
    'quiet' : False,
    'verbose' : False,
    'working_dir' : ''
}

# filenames and patterns -- not changed at runtime

filenames = {\
    'analyzer-tr' : 'analyzer.fsm',
    'config' : 'config.ini',
    'config-default' : 'config-default.ini',
    'eval.graph.vocab' : 'eval-vocab.txt',
    'eval.graph' : 'graph.testing',
    'eval.wordlist' : 'input.testing',
    'eval.report' : 'eval.txt',
    'graph' : 'graph.txt',
    'analyze.graph' : 'graph.analyze',
    'graph-modsel' : 'graph-modsel.txt',
    'index' : 'index.txt',
    'lemmatizer-tr' : 'lemmatizer.fsm',
    'lexicon-tr' : 'lexicon.fsm',
    'log'   : 'log.txt',
    'rootgen-tr' : 'rootgen.fsm',
    'roots-tr' : 'roots.fsm',
    'rules' : 'rules.txt',
    'rules-modsel' : 'rules-modsel.txt',
    'rules-fit' : 'rules-fit.txt',
    'rules-tr' : 'rules.fsm',
    'sample-edge-stats' : 'sample-edge-stats.txt',
    'sample-rule-stats' : 'sample-rule-stats.txt',
    'sample-wordpair-stats' : 'sample-wordpair-stats.txt',
    'tagger-tr' : 'tagger.fsm',
    'wordgen' : 'wordgen.txt',
    'wordlist' : 'input.training',
    'analyze.wordlist' : 'input.analyze',
    'wordlist.left' : 'wordlist-left.training',
    'wordlist.right' : 'wordlist-right.training'
}

format = {\
    'vector_sep' : ' ',
    'word_disamb_sep' : '#',
    'rule_subst_sep' : ':',
    'rule_part_sep' : '/',
    'rule_tag_sep' : '___'
}

patterns = {}
patterns['symbol'] = '(?:[\w\-\.]|\{[A-Z0-9]+\})'
patterns['tag'] = '(?:<[A-Z0-9]+>)'
patterns['disamb'] = '[0-9]+'
patterns['word'] = '^(?P<word>%s+)(?P<tag>%s*)(?:%s(?P<disamb>%s))?$' %\
                          (patterns['symbol'], patterns['tag'], 
                           format['word_disamb_sep'], patterns['disamb'])

patterns['rule_subst'] = '%s*%s%s*' %\
                              (patterns['symbol'], format['rule_subst_sep'], patterns['symbol'])
patterns['rule_named_subst'] = '(?P<x>%s*)%s(?P<y>%s*)' %\
                              (patterns['symbol'], format['rule_subst_sep'], patterns['symbol'])
patterns['rule_tag_subst'] = '%s*%s%s*' %\
                              (patterns['tag'], format['rule_subst_sep'], patterns['tag'])
patterns['rule'] = '^(?P<subst>%s(%s)*)(?:%s(?P<tag_subst>%s))?$' %\
                              (patterns['rule_subst'],
                               format['rule_part_sep']+patterns['rule_subst'],
                               format['rule_tag_sep'],
                               patterns['rule_tag_subst'])
patterns['rule_named_tag_subst'] = '(?P<x>%s*)%s(?P<y>%s*)' %\
                               (patterns['tag'], format['rule_subst_sep'], patterns['tag'])

compiled_patterns = {}
for key, val in patterns.items():
    compiled_patterns[key] = re.compile(val)

normalization_substitutions = {
    '.' : '{FS}',
    '-' : '{HYPH}'
}
unnormalization_substitutions = \
    { val : key for key, val in normalization_substitutions.items() }
multichar_symbols = ['{ALLCAPS}', '{CAP}'] +\
    [subst_to for subst_from, subst_to in normalization_substitutions.items()]

