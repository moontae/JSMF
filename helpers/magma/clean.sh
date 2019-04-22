#!/bin/bash
rm -f run_EXP_$1_exec_$2_K-$3.*
rm -f run_EXP_$1_exec_$2_K-$3_result.*
rm -f EXP_$1_$2_K-$3.log
if [ -z "$4" ]
  then
    rm -f $1_$2_K-$3_*.log
  else
    rm -f $4_$2_K-$3_*.log
fi

