from transformers import AutoTokenizer, AutoModel, FeatureExtractionPipeline
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
import torch
from tqdm.auto import tqdm
import numpy as np
import argparse
import sys
sys.path.insert(0, '.')
from src.utils.nnk_graph import nnk_graph

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', default='./data/test/data.csv')
parser.add_argument('--model', default='bert-base-cased')
parser.add_argument('--tokenizer', default='bert-base-cased')

if __name__ == "__main__":
    args = parser.parse_args()
    
    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else -1
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) 
    model = AutoModel.from_pretrained(args.model)

    dataset = load_dataset("tweet_eval", "sentiment")
    dataset = dataset["test"]
    
    dataset = dataset.filter(lambda example: example["label"] != 1) # remove neutral tweets
    
    features = []
    length = []
    pipe = FeatureExtractionPipeline(model, tokenizer, framweork="pt", return_tensors=True, device=device)
    for embed in tqdm(pipe(KeyDataset(dataset, "text"), batch_size=32), total=len(dataset)):
        # append the embeddings of each token in the tweet
        features.append(embed.squeeze(0))
        length.append(embed.shape[1])

    features = torch.cat(features, dim=0).detach().numpy()

    labels = []
    for i in range(len(dataset)):
        labels += [dataset["label"][i]] * length[i]
    
    tw_indices = []
    for i in range(len(dataset)):
        tw_indices += [i] * length[i]

    weight_values, indices = nnk_graph(features, top_k=50, kernel="ip", cuda_available=cuda_available)
    
    #np.savez_compressed(args.output_file, weight_values=weight_values, indices=indices)
    df = pd.DataFrame({"label": labels, "tw_indices": tw_indices, "weight_values": weight_values.tolist(), "indices": indices.tolist()})
    df.to_csv(args.output_file)




