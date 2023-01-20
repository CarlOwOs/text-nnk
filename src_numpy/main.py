from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np

import torch

from utils.nnk_graph import nnk_graph

if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased")

    dataset = load_dataset("tweet_eval", "sentiment")
    dataset = dataset["train"][0:20]

    # get the text field and labels from the training dataset
    dataset_text = dataset["text"]
    dataset_labels = dataset["label"]
    # list of each tokenized sentence
    dataset_tokens = [tokenizer(sentence, return_tensors="pt") for sentence in dataset_text]
    # get embeddings from sentence tokens and pool the embeddings to get a sentence embedding
    dataset_embeddings = [model(**tokenized_sentence)[0].mean(axis=1) for tokenized_sentence in dataset_tokens]
    features = torch.cat(dataset_embeddings, dim=0).detach().numpy()

    weight_values, error = nnk_graph(features, top_k=5)

        




