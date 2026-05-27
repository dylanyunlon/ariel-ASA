#!/bin/bash

for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=2 python model/geobloom_v19.py --dataset GeoGLUE --epochs 5 | tee -a result/training_geoglue.log
done
