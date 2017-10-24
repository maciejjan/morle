#/bin/bash
BASEDIR=$(dirname $0)

python3 -m unittest discover -t $BASEDIR/src -s $BASEDIR/src/tests
