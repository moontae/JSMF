#!/bin/bash
echo "Universe = Vanilla
Executable = $1_$2_K-$3.sh
Output = $1_$2_K-$3_result.out
Input = /dev/null
Error = $1_$2_K-$3_result.err
Log = $1_$2_K-$3_result.log
Queue 1"
