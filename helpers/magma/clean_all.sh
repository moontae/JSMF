#!/bin/bash
Datasets=('nips_N-5000' 'nytimes_N-15000' 'movies_N-10000' 'songs_N-10000' 'yelp_N-1606' 'blog_N-4447')
for Dataset in "${Datasets[@]}"
do
    ./clean.sh $1 $Dataset
done

