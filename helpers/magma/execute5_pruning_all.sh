#!/bin/bash
Script="run_EXP_pruning_exec"
Datasets=('nips_N-5000' 'nytimes_N-15000' 'movies_N-10000' 'songs_N-10000' 'yelp_N-1606' 'blog_N-4447')
Ps=(0.25 0.50 0.75 1.00)
Ks=(5 10 15 20 25 50 75 100)
InputFolder="/share/magpie/moontae/JSMF"
OutputFolder="/share/magpie/moontae/JSMF"
for Dataset in "${Datasets[@]}"
do
    for P in "${Ps[@]}"
    do
        for K in "${Ks[@]}"
        do  
            ./execute5.sh $Script $Dataset $P $K $InputFolder $OutputFolder
        done
    done
done
for P in "${Ps[@]}"
do  
    ./execute5.sh $Script nytimes_N-15000 $P 125 $InputFolder $OutputFolder
    ./execute5.sh $Script nytimes_N-15000 $P 150 $InputFolder $OutputFolder
done

