from transformers import AutoTokenizer, AutoModel, FeatureExtractionPipeline
from sentence_transformers import SentenceTransformer
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

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', default='./data/test/data.csv')
parser.add_argument('--model', default='bert-base-cased')
parser.add_argument('--tokenizer', default='bert-base-cased')

# given the token embeddings of two sentences, returns the cosine similarity of the minimum distance between the tokens of the two sentences
def sentence_sim_min(embed1, embed2):
    # embed1 and embed2 are of shape (n1, d) and (n2, d)
    # returns a scalar
    n1, n2 = embed1.shape[0], embed2.shape[0]
    embed1, embed2 = embed1.unsqueeze(1).repeat(1, n2, 1), embed2.unsqueeze(0).repeat(n1, 1, 1)
    return torch.max(torch.cosine_similarity(embed1, embed2, dim=2)).item()    

# given the token embeddings of two sentences, returns the cosine similarity of the average minimum distance between the tokens of the two sentences
def sentence_sim_avg_min(embed1, embed2):
    # embed1 and embed2 are of shape (n1, d) and (n2, d)
    # returns a scalar
    n1, n2 = embed1.shape[0], embed2.shape[0]
    embed1, embed2 = embed1.unsqueeze(1).repeat(1, n2, 1), embed2.unsqueeze(0).repeat(n1, 1, 1)
    return torch.mean(torch.max(torch.cosine_similarity(embed1, embed2, dim=2), dim=1).values).item()

# given the embeddings of two sentences, returns their cosine similarity
def sentence_sim(embed1, embed2):
    # embed1 and embed2 are of shape (d)
    return torch.cosine_similarity(embed1, embed2, dim=0).item()

if __name__ == "__main__":
    args = parser.parse_args()
    
    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else -1
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) 
    model = AutoModel.from_pretrained(args.model)
    model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    dataset = load_dataset("stsb_multi_mt", name="en", split="dev")
    dataset = dataset.map(lambda example: {"similarity_score": example["similarity_score"] / 5})
    
    features1 = [] # list of tensors of shape (n1, d)
    features1_avg = [] # list of tensors of shape (1, d)
    pipe = FeatureExtractionPipeline(model, tokenizer, framweork="pt", return_tensors=True, device=device)
    for embed in tqdm(pipe(KeyDataset(dataset, "sentence1"), batch_size=32), total=len(dataset)):
        features1.append(embed)
        features1_avg.append(torch.mean(embed, dim=1))
    
    features2 = [] # list of tensors of shape (n2, d)
    features2_avg = [] # list of tensors of shape (1, d)
    pipe = FeatureExtractionPipeline(model, tokenizer, framweork="pt", return_tensors=True, device=device)
    for embed in tqdm(pipe(KeyDataset(dataset, "sentence2"), batch_size=32), total=len(dataset)):
        features2.append(embed)
        features2_avg.append(torch.mean(embed, dim=1))
        
    features_sbert1 = [] # list of tensors of shape (d)
    for embed in tqdm(model_sbert.encode(dataset["sentence1"], batch_size=32, show_progress_bar=True), total=len(dataset)):
        features_sbert1.append(torch.tensor(embed))
        
    features_sbert2 = [] # list of tensors of shape (d)
    for embed in tqdm(model_sbert.encode(dataset["sentence2"], batch_size=32, show_progress_bar=True), total=len(dataset)):
        features_sbert2.append(torch.tensor(embed))

    pred_min = []
    pred_avg_min = []
    pred = []
    pred_sbert = []
    for i in range(len(dataset)):
        pred_min += [sentence_sim_min(features1[i].squeeze(0), features2[i].squeeze(0))]
        pred_avg_min += [sentence_sim_avg_min(features1[i].squeeze(0), features2[i].squeeze(0))]
        pred += [sentence_sim(features1_avg[i].squeeze(0), features2_avg[i].squeeze(0))]
        pred_sbert += [sentence_sim(features_sbert1[i], features_sbert2[i])]
        
    # give the MSE of each prediction method
    print("MSE of min: ", np.mean(np.square(np.array(pred_min) - np.array(dataset["similarity_score"]))))
    print("MSE of avg_min: ", np.mean(np.square(np.array(pred_avg_min) - np.array(dataset["similarity_score"]))))
    print("MSE of avg tokens: ", np.mean(np.square(np.array(pred) - np.array(dataset["similarity_score"]))))
    print("MSE of SBERT: ", np.mean(np.square(np.array(pred_sbert) - np.array(dataset["similarity_score"]))))
    
    # print spearman rank   
    print("Spearman rank of min: ", spearmanr(pred_min, dataset["similarity_score"])[0]*100)
    print("Spearman rank of avg_min: ", spearmanr(pred_avg_min, dataset["similarity_score"])[0]*100)
    print("Spearman rank of avg tokens: ", spearmanr(pred, dataset["similarity_score"])[0]*100)
    print("Spearman rank of SBERT: ", spearmanr(pred_sbert, dataset["similarity_score"])[0]*100)
    
    # break down the error by similarity score with 10 bins from 0 to 1
    # ...

    # unique_tokens = set()
    # for sentence in dataset["sentence1"]:
    #     unique_tokens.update(tokenizer.tokenize(sentence))
    # for sentence in dataset["sentence2"]:
    #     unique_tokens.update(tokenizer.tokenize(sentence))
    # print("Number of unique tokens:", len(unique_tokens))