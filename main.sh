#!/bin/bash
#data="voc_1over32"
data=$1

## Train a supervised model with labelled images
python3 main.py --config configs/${data}_baseline.json

## Generate class-balanced subclass clusters
python3 main.py --save_feature True --resume saved/${data}_baseline/best_model.pth --config configs/${data}_baseline.json
python3 clustering.py --config configs/${data}_baseline.json --clustering_algorithm balanced_kmeans

## Train a semi-supervised model with both labelled and unlabelled images
python3 main.py --config configs/${data}_usrn.json
