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
