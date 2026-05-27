#!/bin/bash

for i in {1..10}
do
    CUDA_VISIBLE_DEVICES=3 python model/geobloom_v19.py --dataset GeoGLUE_clean --epochs 5 | tee -a result/training_geoglue_clean.log
done
