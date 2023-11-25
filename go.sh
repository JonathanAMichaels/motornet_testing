#!/bin/bash

d1=`date +"%Y-%m-%d %H:%M:%S"`

jupyter nbconvert --to notebook --inplace --execute $1

jupyter nbconvert --to html $1

d2=`date +"%Y-%m-%d %H:%M:%S"`

filename="${1%.ipynb}"
html=".html"
attachment="${filename}${html}"

echo $attachment

python3 notify_email.py $1 "$d1" "$d2" "$attachment"

