#!/bin/bash
Script="run_EXP_$1_exec"
Datasets=('songs_N-10000')
Ks=(5 10)
InputFolder="/share/magpie/moontae/JSMF"
OutputFolder="./models"
for Dataset in "${Datasets[@]}"
do
    for K in "${Ks[@]}"
    do
        ./clean.sh $1 $Dataset $K $2 
        ./execute.sh $Script $Dataset $K $InputFolder $OutputFolder
    done
done

