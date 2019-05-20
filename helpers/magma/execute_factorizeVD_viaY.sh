#!/bin/bash
Script="run_EXP_factorizeVD_viaY_exec"
Datasets=('nips_N-3750' 'nips_N-6250' 'nips_N-7500' 'nips_N-8750' 'movies_N-3750' 'movies_N-6250' 'movies_N-7500' 'movies_N-8750' 'songs_N-2500' 'songs_N-7500' 'songs_N-12500' 'songs_N-15000' 'songs_N-17500' 'blog_N-3000' 'blog_N-4000' 'yelp_N-600' 'yelp_N-1000' 'yelp_N-1200' 'yelp_N-1400' 'yelp_N-1600')
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

Datasets2=('nytimes_N-22500' 'nytimes_N-37500' 'nytimes_N-45000' 'nytimes_N-52500')
Ks2=(5 10 15 20 25 50 75 100 125 150)
for Dataset in "${Datasets2[@]}"
do
    for K in "${Ks2[@]}"
    do
        ./execute.sh $Script $Dataset $K $InputFolder $OutputFolder
    done
done
