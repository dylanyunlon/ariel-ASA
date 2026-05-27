#!/bin/bash

for i in {1..10}
do
    CUDA_VISIBLE_DEVICES=1 python model/geobloom_v19.py --dataset Shanghai | tee -a result/training_shanghai.log
done