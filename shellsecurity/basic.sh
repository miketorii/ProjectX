#!/bin/bash

VAR1='bash security'
VAR2=starting

echo $VAR1
echo $VAR2

VAR3=$(pwd)

echo $VAR3

echo $#
echo $0
echo $1
echo $2
echo $3

echo "Type something:"
read VAR4
echo $VAR4

if cd tmp
then
    echo "here is tmp"
    ls -l
else
    echo "no /tmp"
fi

FILENAME1='test.txt'

if [[ -e $FILENAME1 ]]
then
    echo $FILENAME1 exists
fi
