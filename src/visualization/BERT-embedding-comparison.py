from transformers import AutoTokenizer, AutoModel, FeatureExtractionPipeline
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
import torch
import numpy as np
import argparse
from tqdm.auto import tqdm
import sys
sys.path.insert(0, '.')
import altair as alt

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
    
    dataset = dataset.shuffle(seed=14).select(range(600))

    # get embeddings from sentence tokens
    distances_cls = []
    distances_avg = []
    distances_tokens = []
    
    pipe = FeatureExtractionPipeline(model, tokenizer, framweork="pt", return_tensors=True, device=device)
    for embed in tqdm(pipe(KeyDataset(dataset, "text"), batch_size=32), total=len(dataset)):
        cls_embed = embed[:, 0, :].detach().numpy()
        word_embed = embed[:, 1:, :].squeeze(0).detach().numpy()
        # using inner product distance
        # and also cosine similarity
        # get the distances between the [CLS] token and the other tokens
        distances_cls += [np.inner(cls_embed, word_embed[i]) for i in range(word_embed.shape[0])]
        distances_cls += [np.inner(cls_embed, word_embed[i]) / (np.linalg.norm(cls_embed) * np.linalg.norm(word_embed[i])) for i in range(word_embed.shape[0])]
        # get the distances between the average of the tokens and the other tokens
        distances_avg += [np.inner(np.average(word_embed, axis=0), word_embed[i]) for i in range(word_embed.shape[0])]
        distances_avg += [np.inner(np.average(word_embed, axis=0), word_embed[i]) / (np.linalg.norm(np.average(word_embed, axis=0)) * np.linalg.norm(word_embed[i])) for i in range(word_embed.shape[0])]
        # get the distances between the tokens
        distances_tokens += [np.inner(word_embed[i], word_embed[j]) for i in range(word_embed.shape[0]) for j in range(word_embed.shape[0]) if i != j]
        distances_tokens += [np.inner(word_embed[i], word_embed[j]) / (np.linalg.norm(word_embed[i]) * np.linalg.norm(word_embed[j])) for i in range(word_embed.shape[0]) for j in range(word_embed.shape[0]) if i != j]
    
    # plot the distribution of the distances to cls token
    df = pd.DataFrame({"distances_cls": distances_cls})
    # get the first element of every list
    df["distances_cls"] = df["distances_cls"].apply(lambda x: x[0])
    df["distances_cls"] = pd.cut(df["distances_cls"], bins=100)
    df = df.groupby("distances_cls").size().reset_index(name="count")
    df["distances_cls"] = df["distances_cls"].astype(str)
    chart1 = alt.Chart(df).mark_bar().encode(
        x=alt.X("distances_cls", title="Distance to [CLS] token"),
        y=alt.Y("count", title="Count"),
    ).properties(
        width=300,
        height=300,
        title="Distribution of distances to [CLS] token"
    )
    
    # plot the distribution of the distances to average token
    df = pd.DataFrame({"distances_avg": distances_avg})
    df["distances_avg"] = pd.cut(df["distances_avg"], bins=100)
    df = df.groupby("distances_avg").size().reset_index(name="count")
    df["distances_avg"] = df["distances_avg"].astype(str)
    chart2 = alt.Chart(df).mark_bar().encode(
        x=alt.X("distances_avg", title="Distance to average token"),
        y=alt.Y("count", title="Count"),
    ).properties(
        width=300,
        height=300,
        title="Distribution of distances to average token"
    )

    # plot the distribution of the distances between tokens
    df = pd.DataFrame({"distances_tokens": distances_tokens})
    df["distances_tokens"] = pd.cut(df["distances_tokens"], bins=100)
    df = df.groupby("distances_tokens").size().reset_index(name="count")
    df["distances_tokens"] = df["distances_tokens"].astype(str)
    chart3 = alt.Chart(df).mark_bar().encode(
        x=alt.X("distances_tokens", title="Distance between tokens"),
        y=alt.Y("count", title="Count"),
    ).properties(
        width=300,
        height=300,
        title="Distribution of distances between tokens"
    )

    chart = chart1 | chart2 | chart3
    chart.save(args.output_file)