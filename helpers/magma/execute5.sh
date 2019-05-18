#!/bin/bash
./make5_run.sh $1 $2 $3 $4 $5 $6 > $1_$2_P-$3_K-$4.sh
chmod +x $1_$2_P-$3_K-$4.sh
./make5_sub.sh $1 $2 $3 $4 > $1_$2_P-$3_K-$4.sub
condor_submit $1_$2_P-$3_K-$4.sub
