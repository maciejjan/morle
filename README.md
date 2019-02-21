# Morle -- a Morphemeless Morphology Learner v0.9.0

Morle is a software package for unsupervised learning of language morphology
expressed as string transformations on whole words. It employs a graph
representation of the lexicon, in which words are vertices and words related
via a productive morphological pattern are connected with an edge. Statistical
inference (Monte Carlo Expectation Maximization) is used to learn a model of
morphology from unannotated data. Various tasks can be approached with a
trained model, including clustering of morphologically related words,
predicting new words or predicting POS-tags for unknown words.

As the methods implemented here are subject to ongoing research, the software
might be highly unstable and the user interface is rather rudimentary.

## Installation

Running `python3 setup.py install` will install Morle together with Python 
dependencies. Be aware of the following:

* Python 3 is required (tested on version 3.5.3),
* in addition to HFST Python bindings (which are pulled automatically), Morle
  currently also uses the HFST command-line tools, which have to be installed
  manually following the instructions given [here](hfst.github.io),
* a back-end library for `keras` (either Theano or Tensorflow) has to be
  installed manually (preferably via `pip`); all tests were done on Theano,
* the command `sort` from GNU coreutils is required.

Currently, Morle has only been tested on GNU/Linux systems.

## Usage

Currently, Morle assumes that all files related to a dataset are located in a
single directory (called "working directory") and have fixed names. Each module
assumes the existence of some files that it takes as input and creates other
files in the working directory. A general command to run Morle is:

```
morle MODULE -d WORKING_DIR
```

You can skip the `-d` parameter if working directory is the current directory.

The configuration of the modules is managed on per-experiment basis by the file
`config.ini` in the working directory. If this file does not exist, the default
configuration (to be found in `src/morle/config-default.ini`) is copied to the
working directory on startup.

For details, refer to the documentation of the particular modules. The usual
training workflow is:

```
morle preprocess
morle modsel
morle fit
```

The working directory must contain a list of words (optionally with
frequencies, tab-separated) in a file named `input.training`. You can find
examples of input files and configurations in the `examples/` subdirectory.

## Preferred size of data

Morle works best on datasets between 50k and 300k words. For less than 50k, the
model performance is poor, while lists longer than 300k words lead to high
memory consumption and long training times.

## Author

Maciej Sumalvico <macjan@o2.pl>, NLP Group @ University of Leipzig

## Publications

Maciej Sumalvico, *Unsupervised Learning of Morphology with Graph Sampling*,
RANLP 2017, Varna, Bulgaria, 2017.
