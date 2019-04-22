#!/bin/bash
Datasets=('nips_N-5000' 'nytimes_N-15000' 'movies_N-10000' 'songs_N-10000' 'yelp_N-1606' 'blog_N-4447')
Ks=(5 10 15 20 25 50 75 100)
./clean.sh $1 nytimes_N-15000 125
./clean.sh $1 nytimes_N-15000 150
for Dataset in "${Datasets[@]}"
do
    for K in "${Ks[@]}"
    do
        ./clean.sh $1 $Dataset $K
    done
done

