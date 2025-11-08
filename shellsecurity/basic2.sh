#!/bin/bash

echo "-----------------------------"

ps aux | grep root

wc -m tmp/sample1.txt

echo "-------------------------------"

ps aux | tail

echo "-------------------------------"

ip addr show

echo "-------------------------------"

df

echo "-------------------------------"

awk '$2 == "John"' tmp/sample4.txt

echo "-------------------------------"

awk '$1 == "02"' tmp/sample4.txt

echo "-------------------------------"
echo "---------After Spain-----------"
echo "-------------------------------"

curl google.com

echo "-------------------------------"

if [[ -f /usr/bin/sw_vers ]]; then
    echo "This is macOS"
    echo $OSTYPE
else
    echo "This is NOT macOS"
fi

