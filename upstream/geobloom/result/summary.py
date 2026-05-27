'''
This script is used to summarize the results of the repeated experiments of GeoBloom
We need to report the average and standard deviation of the Recall@20, Recall@10, NDCG@5, and NDCG@1 and the average best dev time, std of best dev time.
'''

import re
import numpy as np

def parse_results(file_content):
    # Regular expression to extract metrics
    pattern = r"Recall@20: (\d+\.\d+)\s+Recall@10: (\d+\.\d+)\s+NDCG@5: (\d+\.\d+)\s+NDCG@1: (\d+\.\d+)\s+Best dev time: (\d+\.\d+)"
    matches = re.findall(pattern, file_content)

    # Convert matches to numpy array for easy calculation
    data = np.array(matches, dtype=float)
    return data

def calculate_statistics(data):
    # Calculate mean and standard deviation
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return means, stds

def summarize_results(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()

    data = parse_results(file_content)
    means, stds = calculate_statistics(data)

    # Print the results
    print("Average Recall@20: {:.6f}, Std: {:.6f}".format(means[0], stds[0]))
    print("Average Recall@10: {:.6f}, Std: {:.6f}".format(means[1], stds[1]))
    print("Average NDCG@5: {:.6f}, Std: {:.6f}".format(means[2], stds[2]))
    print("Average NDCG@1: {:.6f}, Std: {:.6f}".format(means[3], stds[3]))
    
    # Convert average best dev time from seconds to hours, minutes, and seconds
    avg_time_hours = int(means[4] // 3600)
    avg_time_minutes = int((means[4] % 3600) // 60)
    avg_time_seconds = means[4] % 60

    std_time_hours = int(stds[4] // 3600)
    std_time_minutes = int((stds[4] % 3600) // 60)
    std_time_seconds = stds[4] % 60

    print("Average Best dev time: {}h {}m {:.2f}s, Std: {}h {}m {:.2f}s".format(
        avg_time_hours, avg_time_minutes, avg_time_seconds,
        std_time_hours, std_time_minutes, std_time_seconds))

path = 'result/'
dataset_names = ['Beijing', 'Shanghai', 'GeoGLUE', 'GeoGLUE_clean']
version = '19'

for dataset_name in dataset_names:
    file_path = f'{path}{dataset_name}_v{version}_test.txt'
    print(f'Summarizing results for {dataset_name} v{version}')
    summarize_results(file_path)


'''

Summarizing results for Beijing v19
Average Recall@20: 0.780189, Std: 0.002131
Average Recall@10: 0.732639, Std: 0.002173
Average NDCG@5: 0.595475, Std: 0.002075
Average NDCG@1: 0.516190, Std: 0.002760
Average Best dev time: 1h 59m 31.99s, Std: 0h 9m 40.88s
Summarizing results for Shanghai v19
Average Recall@20: 0.840129, Std: 0.001164
Average Recall@10: 0.798841, Std: 0.001867
Average NDCG@5: 0.664742, Std: 0.001978
Average NDCG@1: 0.571568, Std: 0.002994
Average Best dev time: 1h 20m 30.84s, Std: 0h 11m 19.72s
Summarizing results for GeoGLUE v19
Average Recall@20: 0.801262, Std: 0.002741
Average Recall@10: 0.768481, Std: 0.002382
Average NDCG@5: 0.641995, Std: 0.002162
Average NDCG@1: 0.544644, Std: 0.003238
Average Best dev time: 1h 35m 16.57s, Std: 0h 15m 26.68s
Summarizing results for GeoGLUE_clean v19
Average Recall@20: 0.786066, Std: 0.003963
Average Recall@10: 0.731225, Std: 0.003686
Average NDCG@5: 0.551767, Std: 0.004990
Average NDCG@1: 0.423871, Std: 0.007014
Average Best dev time: 0h 26m 42.48s, Std: 0h 5m 39.64s

'''