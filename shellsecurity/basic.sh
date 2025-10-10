#!/bin/bash

function myfunc()
{
    echo "here is in myfunc"
    echo $1 $2 $3
}

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
# read VAR4
# echo $VAR4

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

VAL1=8
echo $VAL1
if (( VAL1 < 12 ))
then
    echo "value $VAL1 is lower than 12"
fi

i=0
while (( i < 5 ))
do
    echo $i
    let i++
done

echo "-----------------------------------"

for (( i=10; i<15; i++ ))
do
    echo $i
done

myfunc
myfunc a b c

echo "-----------------------------------"

grep 'T.o' frost.txt

egrep 'T.?o' frost.txt

egrep '<([A-Za-z]*)>.*</\1>' tags.txt

echo "-----------------------------------"

cut -d' ' -f2 cutfile.txt

file test.png

head test.png

echo "-----------------------------------"

awk '$2 == "Jones" {print $0}' awkusers.txt

find ../../llm -name "*.pth"
find ~ -name ".*"

echo "-----------------------------------"
echo "-----------------------------------"

echo "find by size"
find ../../llm -size +5M

echo "-----------------------------------"

ps

tar -czf my_logs.tar.gz .
mkdir tmptmp
mv my_logs.tar.gz tmptmp
cd tmptmp
tar -xvf my_logs.tar.gz
ls
cd ..
rm -R tmptmp

echo ${HOSTNAME}
echo "-----------------------------------"

#top

w
echo "-----------------------------------"

diff sample1.txt sample2.txt

echo "-----------------------------------"

ps -u

#ps -A
#ps -e

#ps -x

echo "-----------------------------------"
#ps aux | head
ps aux | tail

#ps aux | grep mtorii
#ps aux | grep root



