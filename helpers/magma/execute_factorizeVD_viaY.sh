#!/bin/bash
Script="run_EXP_factorizeVD_viaY_exec"
Datasets=('nips_N-5000' 'movies_N-10000' 'songs_N-10000' 'blog_N-4447' 'yelp_N-1606')
Ks=(5 10 15 20 25 50 75 100)
InputFolder="/share/magpie/moontae/JSMF"
OutputFolder="./models"
for Dataset in "${Datasets[@]}"
do
    for K in "${Ks[@]}"
    do
        ./execute.sh $Script $Dataset $K $InputFolder $OutputFolder
    done
done

Datasets2=('nytimes_N-15000')
Ks2=(5 10 15 20 25 50 75 100 125 150)
for Dataset in "${Datasets2[@]}"
do
    for K in "${Ks2[@]}"
    do
        ./execute.sh $Script $Dataset $K $InputFolder $OutputFolder
    done
done