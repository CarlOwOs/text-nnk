from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import torch
import argparse
import sys
sys.path.insert(0, '.')
from src.utils.nnk_graph import nnk_graph

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', default='./data/test/data.csv')
parser.add_argument('--model', default='bert-base-cased')
parser.add_argument('--dataset', default='tweet_eval.sentiment')

if __name__ == "__main__":
    args = parser.parse_args()
    
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print("Using device:", device)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained(args.model)

    # args_dataset = args.dataset.split(".")
    # if len(args_dataset) == 2:
    #     dataset = load_dataset(args_dataset[0], args_dataset[1])
    # else:
    #     dataset = load_dataset(args_dataset[0])
    #dataset = load_dataset("tweet_eval", "sentiment")
    dataset = load_dataset("amazon_polarity")
    dataset = dataset["test"]
    
    # split the dataset by label and sample
    dataset_negative = dataset.filter(lambda example: example["label"] == 0).shuffle(seed=14).select(range(1000))
    dataset_positive = dataset.filter(lambda example: example["label"] == 1).shuffle(seed=14).select(range(1000))
    # dataset_positive = dataset.filter(lambda example: example["label"] == 2).shuffle(seed=14).select(range(600))
    #dataset = concatenate_datasets([dataset_negative, dataset_neutral, dataset_positive])
    dataset = concatenate_datasets([dataset_negative, dataset_positive])

    # get the text field and labels from the training dataset
    dataset_text = dataset["content"]#["text"]
    dataset_labels = dataset["label"]
    # list of each tokenized sentence
    dataset_tokens = [tokenizer(sentence, return_tensors="pt") for sentence in dataset_text]
    # get embeddings from sentence tokens and pool the embeddings to get a sentence embedding
    dataset_embeddings = [model(**tokenized_sentence)[0].mean(axis=1) for tokenized_sentence in dataset_tokens]
    features = torch.cat(dataset_embeddings, dim=0).detach().numpy()

    weight_values, indices = nnk_graph(features, top_k=50, kernel="ip", cuda_available=cuda_available)
    
    df = pd.DataFrame({"text": dataset_text, "label": dataset_labels, "weight_values": weight_values.tolist(), "indices": indices.tolist()})
    df.to_csv(args.output_file)




