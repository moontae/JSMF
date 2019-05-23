#!/bin/bash
Script="run_EXP_factorizeVD_viaY_vocab_exec"
Datasets=('nips' 'nytimes' 'movies' 'songs' 'blog' 'yelp')
BaseNs=(1250 7500 1250 5000 500 200)
Ks=(5 10 15 20 25 50 75 100)
InputFolder="/share/magpie/moontae/JSMF"
OutputFolder="./models"
numDatasets=${#Datasets[@]}
for ((d=0; d<$numDatasets; d++ ))
do
    Dataset=${Datasets[d]}
    baseN=${BaseNs[d]}
    for ((n=1; n <= 8; n++))
    do
        N=`echo "$baseN * $n" | bc`
        for K in "${Ks[@]}"
        do            
            ./execute.sh $Script ${Dataset}_N-${N} $K $InputFolder $OutputFolder
        done
    done
done
Datasets2=('nytimes' 'songs')
BaseNs2=(7500 5000)
Ks2=(125 150 175 200)
numDatasets2=${#Datasets2[@]}
for ((d=0; d<$numDatasets2; d++ ))
do
    Dataset=${Datasets2[d]}
    baseN=${BaseNs2[d]}
    for ((n=1; n <= 8; n++))
    do
        N=`echo "$baseN * $n" | bc`
        for K in "${Ks2[@]}"
        do            
            ./execute.sh $Script ${Dataset}_N-${N} $K $InputFolder $OutputFolder
        done
    done
done
