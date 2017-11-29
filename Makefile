BASEDIR := $(shell pwd)
PREFIX := /usr

all: test

install:
	sed "s:BASEDIR=.*$$:BASEDIR=$(BASEDIR):" morle > $(PREFIX)/bin/morle
	chmod a+x $(PREFIX)/bin/morle

uninstall:
	rm $(PREFIX)/bin/morle

test:
	python3 -m unittest discover -t ./src -s ./src/tests

typecheck:
	find src/ -name *.py -exec mypy {} \;

pep8-check:
	find src/ -name *.py -exec python3 -m pep8 {} \;
