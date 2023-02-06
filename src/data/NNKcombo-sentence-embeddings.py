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
from src.utils.SingleNNKGraph import SingleNNKGraph

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
    pipe = FeatureExtractionPipeline(model, tokenizer, framweork="pt", return_tensors=True, device=device)
    for embed in tqdm(pipe(KeyDataset(dataset, "text"), batch_size=32), total=len(dataset)):
        cls_embed = embed[:, 0, :].detach().numpy()
        word_embed = embed[:, 1:, :].squeeze(0).detach().numpy()
        weight_values, indices = SingleNNKGraph(word_embed, cls_embed)
        # half weight to nnk, half weight to all
        weight_values = (weight_values + np.ones(len(weight_values))/len(weight_values)) / 2
        sentence_embed = np.average(word_embed, axis=0, weights=weight_values)
        features.append(torch.from_numpy(sentence_embed.astype(np.float32)).unsqueeze(0))

    features = torch.cat(features, dim=0).detach().numpy()    
    weight_values, indices = nnk_graph(features, top_k=50, kernel="ip", cuda_available=cuda_available)
    
    np.savez_compressed(args.output_file, weight_values=weight_values, indices=indices)
    df = pd.DataFrame({"label": dataset["label"], "weight_values": weight_values.tolist(), "indices": indices.tolist()})
    df.to_csv(args.output_file)




