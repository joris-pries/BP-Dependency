#!/bin/bash
for ((i=0; i<=3; i++)) ; do
(
    py all_fi_measures.py $i
) &
  done
wait






