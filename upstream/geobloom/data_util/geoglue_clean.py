'''
This script is used to select the authentic POI for each fake POI in GeoGLUE.
We provide our matching result in the result/sanity_check_ours.txt, which is generated via GPT-4o-turbo in 2023-12-24.
Today's GPT-4o is much better than that. We recommend you to use the latest GPT-4o to generate the geoglue_clean.
NOTE: It may be expensive ($100+?) to run this script! Use it at your own risk!
'''
import os
import heapq
import re

import jieba_fast as jieba

from openai import OpenAI
from tqdm import tqdm
from collections import OrderedDict
from scipy.spatial import cKDTree

# Set the OpenAI API key
if os.path.exists('openai_api_key.txt'):
    with open('openai_api_key.txt', 'r', encoding='utf-8') as f:
        OPENAI_API_KEY = f.read().strip()
else:
    OPENAI_API_KEY = None

assert OPENAI_API_KEY is not None
client = OpenAI(api_key=OPENAI_API_KEY)

fake_poi_text = []
fake_poi_location = []

# Load the fake POI data in GeoGLUE
with open('data/GeoGLUE/poi.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm(lines, desc='Loading fake POI'):
        line = line.strip().split('\t')
        fake_poi_text.append(line[0])
        fake_poi_location.append([float(line[1]), float(line[2])])

train_queries = []
train_query_truth_dict = OrderedDict()
dev_queries = []
dev_query_truth_dict = OrderedDict()
test_queries = []
test_query_truth_dict = OrderedDict()

# Load the query data in GeoGLUE
with open('data/GeoGLUE/train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line_id, line in enumerate(tqdm(lines, desc='Loading train queries')):
        fields = line.strip().split('\t')
        assert len(fields) == 4
        line_without_truth = '\t'.join(fields[:3])
        truth_id = int(fields[3]) # GeoGLUE has only one truth id for each query
        train_queries.append((line_without_truth, truth_id))
        train_query_truth_dict[truth_id] = line_id

with open('data/GeoGLUE/dev.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line_id, line in enumerate(tqdm(lines, desc='Loading dev queries')):
        fields = line.strip().split('\t')
        assert len(fields) == 4
        truth_id = int(fields[3]) # GeoGLUE has only one truth id for each query
        dev_queries.append((line_without_truth, truth_id))
        dev_query_truth_dict[truth_id] = line_id

with open('data/GeoGLUE/test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line_id, line in enumerate(tqdm(lines, desc='Loading test queries')):
        fields = line.strip().split('\t')
        assert len(fields) == 4 
        truth_id = int(fields[3]) # GeoGLUE has only one truth id for each query
        test_queries.append((line_without_truth, truth_id))
        test_query_truth_dict[truth_id] = line_id

all_truth = set(train_query_truth_dict.keys()) | set(dev_query_truth_dict.keys()) | set(test_query_truth_dict.keys())

print(f'Total {len(all_truth)} truth POIs')

poi_text = []
poi_cut_text = []
poi_location = []

# Load the authentic POI data
with open('data/GeoGLUE_clean/poi.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm(lines, desc='Loading Authentic POIs'):
        line = line.strip().split('\t')
        poi_text.append(line[0])
        poi_cut_text.append(jieba.lcut_for_search(line[0]))
        poi_location.append([float(line[1]), float(line[2])])

# Build the KDTree for the authentic POI data
poi_tree = cKDTree(poi_location)

candidate_dict = OrderedDict()

# The GeoGLUE POI data is randomly shifted by 1000m, so we set the radius to 2000m to fully cover the shifted area
# We use the KDTree to find all possible candidates < 2000m for each fake POI
for truth_id in tqdm(all_truth, desc='Searching candidate authentic POIs'):
    truth_text = fake_poi_text[truth_id]
    truth_lat = fake_poi_location[truth_id][0]
    truth_lon = fake_poi_location[truth_id][1]
    truth_cut_text = set(jieba.lcut_for_search(truth_text))
    candidate_index = poi_tree.query_ball_point([truth_lat, truth_lon], 2000, return_sorted=True, workers=32)
    if len(candidate_index) == 0:
        continue
    candidate_text = [poi_cut_text[i] for i in candidate_index]
    scores = [len(truth_cut_text.intersection(text)) for text in candidate_text]
    # select the top 10 candidates's index and text
    top_num = min(10, len(scores))
    top_index = heapq.nlargest(top_num, range(len(scores)), key=lambda x: scores[x])
    top_candidate_index = [candidate_index[i] for i in top_index]
    top_candidate_text = [poi_text[i] for i in top_candidate_index]
    # top_scores = [scores[i] for i in top_index]
    candidate_dict[truth_id] = {
        'index': top_candidate_index,
        'text': top_candidate_text,
        'GPT-4o-selected': -1
    }

# sort the candidate_dict by keys
candidate_dict = OrderedDict(sorted(candidate_dict.items(), key=lambda x: x[0]))

# We now use GPT-4o to select the authentic POI for each fake POI
# If the GPT-4o can't find any authentic POI in the top 20, we regard the POI as a noise and ignore it.

system_prompt = "You are an assistant for Geospatial Entity Resolution. Given a vague POI (either fake or authentic) and a list of indexed candidate authentic POIs, identify the POI that most closely matches the vague POI. Return 'Index: number' for the best match, or 'Index: -1' if no match is found."

# Function to query OpenAI for POI selection
def select_poi_with_gpt4o(truth_id, candidate_info, num_candidate):
    # Construct the message for the API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Vague POI: {fake_poi_text[truth_id]}\nCandidates: {candidate_info}"}
    ]
    
    # Make the API call
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,  # Adjust as needed
        temperature=0.2  # Adjust as needed for more deterministic responses
    )
    
    content = completion.choices[0].message.content.strip()
    # Extract the number following "Index:" from the response using regex
    match = re.search(r'Index:\s*(-?\d+)', content)
    if match:
        selected_index = int(match.group(1))
        if selected_index < 0:
            print(f'GPT-4o found no candidate for {fake_poi_text[truth_id]}.')
        elif selected_index >= num_candidate:
            print(f'GPT-4o found a candidate out of range for {fake_poi_text[truth_id]}.')
            print(f'GPT-4o output: {content}')
            selected_index = -1
    else:
        if content.endswith('-1'):
            # Sometimes GPT-4o will output -1 and ignore Index:
            print(f'GPT-4o found no candidate for {fake_poi_text[truth_id]}.')
        else:
            print(f'Error: GPT-4o output no number: {content}')
            selected_index = -1
    
    return selected_index

checked_truth_ids = set()
# load the previous result if exists
if os.path.exists('data_util/sanity_check.txt'):
    with open('data_util/sanity_check.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            checked_truth_ids.add(int(line[0]))

selected_num = len(checked_truth_ids)

with open('data_util/sanity_check.txt', 'w+', encoding='utf-8') as f:
    for truth_id, candidate in tqdm(candidate_dict.items(), desc='Selecting authentic POIs via GPT-4o'):
        if truth_id in checked_truth_ids:
            continue
        candidate_info = ", ".join([f"[Index: {idx}, Text: {text}]" for idx, text in enumerate(candidate['text'])])
        try:
            selected_index = select_poi_with_gpt4o(truth_id, candidate_info, num_candidate=len(candidate['text']))
        except Exception as e:
            print(f'Error: {e} when selecting POI for {fake_poi_text[truth_id]}.')
            selected_index = -1

        if selected_index != -1:
            real_index = candidate['index'][selected_index]
            candidate_dict[truth_id]['GPT-4o-selected'] = real_index
            selected_num += 1
            f.write(f'{truth_id}\t{fake_poi_text[truth_id]}\t{poi_text[real_index]}\n')

print(f'{selected_num}/{len(all_truth)} fake POIs have found authentic POIs with GPT-4o')
# iterate and save the queries with the selected POI index
os.makedirs('data/GeoGLUE_clean/tmp/', exist_ok=True)

with open('data/GeoGLUE_clean/tmp/train.txt', 'w', encoding='utf-8') as f:
    for line_without_truth, old_truth_id in train_queries:
        new_truth_id = candidate_dict[old_truth_id]['GPT-4o-selected']
        if new_truth_id != -1:
            new_line = f'{line_without_truth}\t{new_truth_id}\n'
            f.write(new_line)

with open('data/GeoGLUE_clean/tmp/dev.txt', 'w', encoding='utf-8') as f:
    for line_without_truth, old_truth_id in dev_queries:
        new_truth_id = candidate_dict[old_truth_id]['GPT-4o-selected']
        if new_truth_id != -1:
            new_line = f'{line_without_truth}\t{new_truth_id}\n'
            f.write(new_line)

with open('data/GeoGLUE_clean/tmp/test.txt', 'w', encoding='utf-8') as f:
    for line_without_truth, old_truth_id in test_queries:
        new_truth_id = candidate_dict[old_truth_id]['GPT-4o-selected']
        if new_truth_id != -1:
            new_line = f'{line_without_truth}\t{new_truth_id}\n'
            f.write(new_line)

