#!/bin/bash
./make_run.sh $1 $2 $3 $4 $5 > $1_$2_K-$3.sh
chmod +x $1_$2_K-$3.sh
./make_sub.sh $1 $2 $3 > $1_$2_K-$3.sub
condor_submit $1_$2_K-$3.sub
