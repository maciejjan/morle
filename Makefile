all: typecheck

typecheck:
	find src/ -name *.py -exec mypy {} \;

pep8-check:
	find src/ -name *.py -exec python3 -m pep8 {} \;
