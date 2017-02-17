#!/bin/bash

while read -r line; do
  set -- $line
  pdf2svg results.pdf $6.svg $1
  page=$1
  echo $6.svg created
done < <(cat -n rawdataList)

while read -r line; do
  set -- $line
  page=$((page + 1))
  pdf2svg results.pdf $1.svg $page
  echo $1.svg created
done < <(cat cmpList)

while read -r line; do
  set -- $line
  page=$((page + 1))
  pdf2svg results.pdf $1.svg $page
  echo $1.svg created
done < <(cat distrList)
