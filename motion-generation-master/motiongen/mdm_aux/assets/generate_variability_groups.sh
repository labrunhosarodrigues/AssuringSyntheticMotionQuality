#!/bin/bash
SOURCE=$1

selection=''
inputs=''
for s in {0..5}; do \
  echo "sample "$s >> ${SOURCE}/shuffling.txt;
  for r in {0..6}; do \
    inputs='';
    selection='';
    for i in $(shuf -i 0-9 -n 5); do \
      inputs+="-i "${SOURCE}"/sample0"${s}"_rep0"${i}".mp4 ";
      selection+=$i" ";
    done;
    echo $selection >> ${SOURCE}/shuffling.txt;
    ffmpeg ${inputs} \
      -filter-complex hstack=inputs=5 \
      ${SOURCE}/sample0${s}_group0${r}.mp4;
  done;
done;