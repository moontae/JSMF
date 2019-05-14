#!/bin/bash
Script="run_EXP_factorizeC_viaY_exec"
Datasets=('nips_N-5000' 'nytimes_N-15000' 'movies_N-10000' 'songs_N-10000' 'yelp_N-1606' 'blog_N-4447')
Ks=(5 10 15 20 25 50 75 100)
InputFolder="/share/magpie/moontae/JSMF"
OutputFolder="./models"
./execute.sh $Script nytimes_N-15000 125 $InputFolder $OutputFolder
./execute.sh $Script nytimes_N-15000 150 $InputFolder $OutputFolder
for Dataset in "${Datasets[@]}"
do
    for K in "${Ks[@]}"
    do
        ./execute.sh $Script $Dataset $K $InputFolder $OutputFolder
    done
done

