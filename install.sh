#!/bin/bash

BASEDIR=$(dirname $(realpath $0))
PREFIX=/usr

for i in "$@"
do
case $i in
	--prefix=*)
	PREFIX="${i#*=}"
	shift
	;;
	*)
	;;
esac
done

sed "s:BASEDIR=.*$:BASEDIR=$BASEDIR:" $BASEDIR/morle > $PREFIX/bin/morle
chmod a+x $PREFIX/bin/morle

