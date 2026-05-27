import os
import random
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='GeoGLUE_clean')

args = argparser.parse_args()
dataset = args.dataset  

dataset_path = 'data/' + dataset + '/'
train_path = dataset_path + 'train.txt'
portion_path = dataset_path + 'portion/'

if not os.path.exists(portion_path):
    os.makedirs(portion_path)

# We randomly pick 2%, 5%, 10%, 30%, 50%, 70% of the training data as the training portion

portions = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


with open(train_path, 'r') as f:
    lines = f.readlines()
    total_lines = len(lines)

# Fix the random seed to make all results reproducible
random.seed(0)
random.shuffle(lines)

for portion in portions:
    portion_lines = int(total_lines * portion)
    portion_file = portion_path + 'train_' + str(portion) + '.txt'
    with open(portion_file, 'w') as f:
        for line in lines[:portion_lines]:
            f.write(line)

print('Training portions are saved in ' + portion_path)

