import csv
from tqdm import tqdm
import os
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    BertTokenizer,
    BertModel,
    )
import torch
import numpy as np
from accelerate import PartialState

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",default="GeoGLUE")
    parser.add_argument("--task", default="poi")
    parser.add_argument("--encoding_batch_size",type=int,default=1024)
    parser.add_argument("--step",type=int,default=320)
    # parser.add_argument("--poi_encoder_path",default="wandb/latest-run/files/step-4278/doc_encoder")
    # parser.add_argument("--output_dir",default="embeddings/Beijing/")

    args = parser.parse_args()
    dataset = args.dataset
    # pick the step with the best (lowest) avg_rank on the dev set
    step = args.step

    output_dir = f'embeddings/{dataset}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task = args.task
    assert task in ["poi", "query"]

    if task == "poi":
        encoder_path = f"wandb/latest-run/files/step-{step}/doc_encoder"
        path = f'../../data/{dataset}/poi.txt'
    else:
        encoder_path = f"wandb/latest-run/files/step-{step}/query_encoder"
        path = f'../../data/{dataset}/test.txt'

    distributed_state = PartialState()
    device = distributed_state.device

    ## load encoder
    doc_encoder = BertModel.from_pretrained(encoder_path,add_pooling_layer=False)
    tokenizer = BertTokenizer.from_pretrained(encoder_path)
    doc_encoder.eval()
    doc_encoder.to(device)

    docs = []
    with open(path) as f:
        for line in f:
            docs.append(line.strip().split('\t')[0])

    with distributed_state.split_between_processes(docs) as sharded_docs:
        sharded_docs = [sharded_docs[idx:idx+args.encoding_batch_size] for idx in range(0,len(sharded_docs),args.encoding_batch_size)]
        encoding_progress_bar = tqdm(total=len(sharded_docs), disable=not distributed_state.is_main_process,ncols=100,desc=f'encoding {task}...')
        doc_embeddings = []
        for data in sharded_docs:
            model_input = tokenizer(data,max_length=256,padding='max_length',return_tensors='pt',truncation=True).to(device)
            with torch.no_grad():
                if isinstance(doc_encoder,BertModel):
                    CLS_POS = 0
                    output = doc_encoder(**model_input).last_hidden_state[:,CLS_POS,:].cpu().numpy()
                else:
                    output = doc_encoder(**model_input).pooler_output.cpu().numpy()
            doc_embeddings.append(output)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings,axis=0)
        os.makedirs(output_dir,exist_ok=True)
        np.save(f'{output_dir}/{task}_{distributed_state.process_index}.npy',doc_embeddings)


