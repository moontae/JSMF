#!/bin/bash
Script="run_EXP_FactorizeVD_viaY_exec"
Datasets=('nips_N-1250' 'nips_N-2500' 'nips_N-5000' 'nips_N-10000' 'movies_N-1250' 'movies_N-2500' 'movies_N-5000' 'movies_N-10000' 'songs_N-5000' 'songs_N-10000' 'songs_N-20000' 'songs_N-40000' 'blog_N-500' 'blog_N-1000' 'blog_N-2000' 'blog_N-4447' 'yelp_N-200' 'yelp_N-400' 'yelp_N-800' 'yelp_N-1606')
Ks=(5 10 15 20 25 50 75 100)
InputFolder="/share/magpie/moontae/JSMF"
OutputFolder="./models"
for Dataset in "${Datasets[@]}"
do
    for K in "${Ks[@]}"
    do
        Filename=../../experiments/EXP_FactorizeVD_viaY/${Script}_${Dataset}_K-${K}_result.err
        Filesize=$(stat -c%s "$Filename")
        if [ ${Filesize} -gt 0 ]
        then
            echo "$Filesize bytes ($Filename)"
            ./execute.sh $Script $Dataset $K $InputFolder $OutputFolder
        fi        
    done
done

Datasets2=('nytimes_N-7500' 'nytimes_N-15000' 'nytimes_N-30000' 'nytimes_N-60000')
Ks2=(5 10 15 20 25 50 75 100 125 150)
for Dataset in "${Datasets2[@]}"
do
    for K in "${Ks2[@]}"
    do
        Filename=../../experiments/EXP_FactorizeVD_viaY/${Script}_${Dataset}_K-${K}_result.err
        Filesize=$(stat -c%s "$Filename")
        if [ ${Filesize} -gt 0 ]
        then
            echo "$Filesize bytes ($Filename)"
            ./execute.sh $Script $Dataset $K $InputFolder $OutputFolder
        fi    
    done
done