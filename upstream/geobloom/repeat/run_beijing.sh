#!/bin/bash

for i in {1..10}
do
    CUDA_VISIBLE_DEVICES=0 python model/geobloom_v19.py --dataset Beijing | tee -a result/training_beijing.log
done
