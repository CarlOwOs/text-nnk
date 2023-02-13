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
from scipy.stats import spearmanr
from scipy.optimize import lsq_linear
import altair as alt

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', default='./data/test/data.csv')
parser.add_argument('--model', default='bert-base-cased')
parser.add_argument('--tokenizer', default='bert-base-cased')

# given the token embeddings of two sentences, returns the similarity based on the reconstruction error of the tokens of sentence 1 wrt sentence 2
def sentenceSimAB(embed1, embed2):
    embed1 = embed1.double().detach().cpu()
    embed2 = embed2.double().detach().cpu()
    
    weights = []
    for i in range(embed1.shape[0]):
        res = lsq_linear(embed2.transpose(1, 0), embed1[i], bounds=(0, 1))
        weights.append(res.x)
    weights = torch.tensor(np.array(weights))
    
    err = torch.sum(torch.norm(embed1 - torch.matmul(weights, embed2), dim=1)).item()
    # err = torch.sum(1-torch.cosine_similarity(embed1, torch.matmul(weights, embed2), dim=1)).item()
    err_avg = err / embed1.shape[0]
    
    return 1/err, 1/err_avg

# given the token embeddings of two sentences, returns the cosine similarity of the minimum distance between the tokens of the two sentences
def sentenceSimMinS(embed1, embed2):
    n1, n2 = embed1.shape[0], embed2.shape[0]
    embed1, embed2 = embed1.unsqueeze(1).repeat(1, n2, 1), embed2.unsqueeze(0).repeat(n1, 1, 1)
    return torch.max(torch.cosine_similarity(embed1, embed2, dim=2)).item()    

# given the token embeddings of two sentences, returns the cosine similarity of the average minimum distance between the tokens of the two sentences
def sentenceSimAvgMinS(embed1, embed2):
    n1, n2 = embed1.shape[0], embed2.shape[0]
    embed1, embed2 = embed1.unsqueeze(1).repeat(1, n2, 1), embed2.unsqueeze(0).repeat(n1, 1, 1)
    return torch.mean(torch.max(torch.cosine_similarity(embed1, embed2, dim=2), dim=1).values).item()

def sentenceSimAvg(embed1, embed2):
    embed1 = torch.mean(embed1, dim=0)
    embed2 = torch.mean(embed2, dim=0)
    return torch.cosine_similarity(embed1, embed2, dim=0).item()

if __name__ == "__main__":
    args = parser.parse_args()
    
    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else -1
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) 
    model = AutoModel.from_pretrained(args.model)

    dataset = load_dataset("stsb_multi_mt", name="en", split="dev")
    
    features1 = []
    # pipe = FeatureExtractionPipeline(model, tokenizer, framweork="pt", return_tensors=True, device=device)
    # for embed in tqdm(pipe(KeyDataset(dataset, "sentence1"), batch_size=32), total=len(dataset)):
    # for i in tqdm(range(len(dataset)), total=len(dataset)):
    #     embed = model(**tokenizer(dataset["sentence1"][i], return_tensors="pt"), output_hidden_states=True).hidden_states
    #     features1.append(embed)
    # torch.save(features1, "data/features1.pt")
    features1 = torch.load("data/features1.pt")
    
    features2 = []
    # pipe = FeatureExtractionPipeline(model, tokenizer, framweork="pt", return_tensors=True, device=device)
    # for embed in tqdm(pipe(KeyDataset(dataset, "sentence2"), batch_size=32), total=len(dataset)):
    # for i in tqdm(range(len(dataset)), total=len(dataset)):
    #     embed = model(**tokenizer(dataset["sentence2"][i], return_tensors="pt"), output_hidden_states=True).hidden_states
    #     features2.append(embed)
    # torch.save(features2, "data/features2.pt")
    features2 = torch.load("data/features2.pt")
    
    # embed[0] is the initial embedding
    # embed[-1] is the las_hidden_state
    # embed[i, 1:-1, :] excludes the [CLS] and [SEP] tokens
    
    predABSum, predABAvg, predMinS, predAvgMinS, predAvg = [], [], [], [], []
    for j in tqdm(range(len(dataset)), total=len(dataset)):
        # average the correspontind element of each tuple
        embed1 = torch.mean(torch.stack([features1[j][i][:,1:-1, :].squeeze(0) for i in range(13)], dim=0), dim=0)
        embed2 = torch.mean(torch.stack([features2[j][i][:,1:-1, :].squeeze(0) for i in range(13)], dim=0), dim=0)

        ab_sum, ab_avg = sentenceSimAB(embed1, embed2)
        predABSum.append(ab_sum)
        predABAvg.append(ab_avg)
        predMinS.append(sentenceSimMinS(embed1, embed2))
        predAvgMinS.append(sentenceSimAvgMinS(embed1, embed2))
        predAvg.append(sentenceSimAvg(embed1, embed2))
        
    resultsABSum = spearmanr(dataset["similarity_score"], predABSum)[0]*100
    resultsABAvg = spearmanr(dataset["similarity_score"], predABAvg)[0]*100
    resultsMinS = spearmanr(dataset["similarity_score"], predMinS)[0]*100
    resultsAvgMinS = spearmanr(dataset["similarity_score"], predAvgMinS)[0]*100
    resultsAvg = spearmanr(dataset["similarity_score"], predAvg)[0]*100
    
    results = pd.DataFrame({"ABSum": [resultsABSum], "ABAvg": [resultsABAvg], "MinS": [resultsMinS], "AvgMinS": [resultsAvgMinS], "Avg": [resultsAvg]})
    results.to_csv("data/results-layers-aggr.csv")