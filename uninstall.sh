#!/bin/bash

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

rm $PREFIX/bin/morle

