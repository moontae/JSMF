#!/bin/bash
Script="run_EXP_factorizeC_viaY_exec"
Datasets=('nips_N-5000' 'nytimes_N-15000' 'movies_N-10000' 'songs_N-10000' 'yelp_N-1606' 'blog_N-4447')
Ks=(5 10 15 20 25 50 75 100)
InputFolder="/share/magpie/moontae/JSMF"
OutputFolder="./models"
for Dataset in "${Datasets[@]}"
do
    for K in "${Ks[@]}"
    do
        Filename=../../experiments/EXP_factorizeC_viaY/${Script}_${Dataset}_K-${K}_result.err
        Filesize=$(stat -c%s "$Filename")
        if [ ${Filesize} -gt 0 ]
        then
            echo "$Filesize bytes ($Filename)"
            ./execute.sh $Script $Dataset $K $InputFolder $OutputFolder
        fi        
    done
done

Ks2=(125 150)
for K in "${Ks2[@]}"
do
    Filename=../../experiments/EXP_factorizeC_viaY/${Script}_${Dataset}_K-${K}_result.err
    Filesize=$(stat -c%s "$Filename")
    if [ ${Filesize} -gt 0 ]
    then
        echo "$Filesize bytes ($Filename)"
        ./execute.sh $Script nytimes_N-15000 $K $InputFolder $OutputFolder
    fi    
done
