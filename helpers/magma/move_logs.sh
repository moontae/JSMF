#!/bin/bash
Datasets=('nips_N-5000' 'nytimes_N-15000' 'movies_N-10000' 'songs_N-10000' 'yelp_N-1606' 'blog_N-4447')
OutputFolder="./models"
Suffix="AP"
for Dataset in "${Datasets[@]}"
do
    LogFolder="${OutputFolder}/${Dataset}_${Suffix}/log"
    mkdir -p $LogFolder
    mv ${1}_${Dataset}_K-*.log $LogFolder
    mv EXP_${1}_${Dataset}_K-*.log $LogFolder
    cp run_EXP_${1}_exec_${Dataset}_K-*_result.out $LogFolder
done

