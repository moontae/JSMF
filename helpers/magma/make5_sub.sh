#!/bin/bash
echo "Universe = Vanilla
Executable = $1_$2_P-$3_K-$4.sh
Output = $1_$2_P-$3_K-$4_result.out
Input = /dev/null
Error = $1_$2_P-$3_K-$4_result.err
Log = $1_$2_P-$3_K-$4_result.log
Queue 1"
