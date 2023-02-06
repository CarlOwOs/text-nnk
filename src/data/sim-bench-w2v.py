from datasets import load_dataset
import pandas as pd
import time
import argparse
import sys
import torch
sys.path.insert(0, '.')
import io
from gensim.models import KeyedVectors
from gensim.utils import tokenize
import numpy as np
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', default='./data/test/data.csv')

def sentence_sim_min(embed1, embed2):
    # embed1 and embed2 are of shape (n1, d) and (n2, d)
    # returns a scalar
    n1, n2 = embed1.shape[0], embed2.shape[0]
    embed1, embed2 = embed1.unsqueeze(1).repeat(1, n2, 1), embed2.unsqueeze(0).repeat(n1, 1, 1)
    return torch.max(torch.cosine_similarity(embed1, embed2, dim=2)).item()

def sentence_sim_avg_min(embed1, embed2):
    # embed1 and embed2 are of shape (n1, d) and (n2, d)
    # returns a scalar
    n1, n2 = embed1.shape[0], embed2.shape[0]
    embed1, embed2 = embed1.unsqueeze(1).repeat(1, n2, 1), embed2.unsqueeze(0).repeat(n1, 1, 1)
    return torch.mean(torch.max(torch.cosine_similarity(embed1, embed2, dim=2), dim=1).values).item()

def sentence_sim(embed1, embed2):
    # embed1 and embed2 are of shape (d)
    return torch.cosine_similarity(embed1, embed2, dim=0).item()

if __name__ == "__main__":
    args = parser.parse_args()
    
    dataset = load_dataset("stsb_multi_mt", name="en", split="dev")
    dataset = dataset.map(lambda example: {"similarity_score": example["similarity_score"] / 5})
    
    start = time.time()
    model = KeyedVectors.load_word2vec_format('./models/wiki-news-300d-1M.vec', binary=False)
    
    features1_w2v = []
    features1_w2v_avg = []
    for tweet in dataset['sentence1']:
        tweet = tokenize(tweet)
        tweet = [word for word in tweet if word in model]
        tweet = torch.tensor([model[word] for word in tweet])
        features1_w2v.append(tweet)
        features1_w2v_avg.append(torch.mean(tweet, dim=0))
        
    features2_w2v = []
    features2_w2v_avg = []
    for tweet in dataset['sentence2']:
        tweet = tokenize(tweet)
        tweet = [word for word in tweet if word in model]
        tweet = torch.tensor([model[word] for word in tweet])
        features2_w2v.append(tweet)
        features2_w2v_avg.append(torch.mean(tweet, dim=0))

    pred_min = []
    pred_avg_min = []
    pred = []
    for i in range(len(dataset)):
        pred_min += [sentence_sim_min(features1_w2v[i], features2_w2v[i])]
        pred_avg_min += [sentence_sim_avg_min(features1_w2v[i], features2_w2v[i])]
        pred += [sentence_sim(features1_w2v_avg[i], features2_w2v_avg[i])]
    
    end = time.time()
    print("Time taken: ", end - start)
    
    # print the results MSE
    print("MSE of min: ", np.mean(np.square(np.array(pred_min) - np.array(dataset["similarity_score"]))))
    print("MSE of avg_min: ", np.mean(np.square(np.array(pred_avg_min) - np.array(dataset["similarity_score"]))))
    print("MSE of avg tokens: ", np.mean(np.square(np.array(pred) - np.array(dataset["similarity_score"]))))
    
    # print the results Spearman's rank
    print("Spearman's rank of min: ", spearmanr(pred_min, dataset["similarity_score"])[0]*100)
    print("Spearman's rank of avg_min: ", spearmanr(pred_avg_min, dataset["similarity_score"])[0]*100)
    print("Spearman's rank of avg tokens: ", spearmanr(pred, dataset["similarity_score"])[0]*100)

    
    # unique_tokens = set()
    # for tweet in dataset['sentence1']:
    #     tweet = tokenize(tweet)
    #     unique_tokens.update(tweet)
    # for tweet in dataset['sentence2']:
    #     tweet = tokenize(tweet)
    #     unique_tokens.update(tweet)
    # print("Number of unique tokens: ", len(unique_tokens))