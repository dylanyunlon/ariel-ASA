import os
import numpy as np

def fast_ndcg(prediction, truth, k):
    '''
        This fast_ndcg regard all labeled queries as equally important,
        As the Beijing and Shanghai data is provided in this way (The user clicks as no order),
        and the GeoGLUE dataset has only 1 truth for each query.
    '''
    # prediction: list of indices
    # truth: list of indices. 
    # k: int
    # return: float
    dcg = 0
    for i in range(min(len(prediction), k)):
        if prediction[i] in truth:
            dcg += 1 / np.log2(i + 2)
    idcg = 0
    for i in range(min(len(truth), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg

def recall(prediction, truth, k):
    # prediction: list of indices
    # truth: list of indices
    # k: int
    # return: float
    return len(set(prediction[:k]).intersection(set(truth))) / min(len(set(truth)), k)

# Not used in the paper
def mrr(prediction, truth, k):
    # prediction: list of indices
    # truth: list of indices
    # k: int
    # return: float
    for i in range(min(len(prediction), k)):
        if prediction[i] in truth:
            return 1 / (i + 1)
    return 0

def evaluate(top100, truths, metrics):
    # top100: list of list of indices
    # truths: list of list of indices
    # metrics: dict of metrics
    # return: dict of metric values
    results = {}
    for metric_name, metric_func in metrics.items():
        results[metric_name] = np.mean([metric_func(p, t) for p, t in zip(top100, truths)])
    return results

if __name__ == '__main__':

    datasets = [
        # 'Beijing',
        # 'Shanghai',
        # 'GeoGLUE',
        # 'GeoGLUE_clean',
        'Synthetic',
    ]
    metrics = {
        # 'Recall @ 1000': lambda p, t: recall(p, t, 1000),
        'Recall @ 20': lambda p, t: recall(p, t, 20),
        'Recall @ 10': lambda p, t: recall(p, t, 10),
        'NDCG @ 5': lambda p, t: fast_ndcg(p, t, 5),
        'NDCG @ 1': lambda p, t: fast_ndcg(p, t, 1),
    }
    models = [
        # 'BM25',
        'BM25_D',
        # 'BERT',
        # 'BERT_D',
        # 'OpenAI',
        # 'OpenAI_D',
        # 'DPR',
        # 'DPR_D',
    ]

    portions = [
        0.02, 0.05, 0.1, 0.3, 0.5, 0.7
    ]
    for dataset in datasets:
        query_truth = []
        with open(f'data/{dataset}/test.txt', 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                query_truth_str = line[3]
                query_truth_split = query_truth_str.split(',')
                query_truth.append([int(x) for x in query_truth_split])
        for model in models:

            def test(result_file):
                if not os.path.exists(result_file):
                    return
                top100 = np.load(result_file)
                print(f'{dataset}\t{model}\tlength={top100.shape[-1]}\n', end='')
                results = evaluate(top100, query_truth, metrics)
                result_string = ''
                for _, metric_value in results.items():
                    result_string += f'{metric_value}\t'
                print(result_string[:-1])

            test(f'result/{dataset}_{model}_top100.npy')
            for portion in portions:
                test(f'result/{dataset}_{model}_{portion}_top100.npy')
                


    