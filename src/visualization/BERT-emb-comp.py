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
    
    dataset = dataset.shuffle(seed=14).select(range(20))

    dataset_tokens = [tokenizer(sentence, return_tensors="pt") for sentence in dataset["text"]]
    print(dataset["text"][0])
    print(tokenizer.convert_ids_to_tokens(tokenizer.encode(dataset["text"][0])))
    for tokenized_sentence in dataset_tokens:
        embed = model(**tokenized_sentence)

        cls_embed = embed[0][:, 0, :].detach().numpy()
        word_embed = embed[0][:, 1:, :].squeeze(0).detach().numpy()

        ip_cls = [np.inner(cls_embed, word_embed[i]) for i in range(word_embed.shape[0])]
        cos_cls = [np.inner(cls_embed, word_embed[i]) / (np.linalg.norm(cls_embed) * np.linalg.norm(word_embed[i])) for i in range(word_embed.shape[0])]
        print(cos_cls)
        ip_avg = [np.inner(np.average(word_embed, axis=0), word_embed[i]) for i in range(word_embed.shape[0])]
        cos_avg = [np.inner(np.average(word_embed, axis=0), word_embed[i]) / (np.linalg.norm(np.average(word_embed, axis=0)) * np.linalg.norm(word_embed[i])) for i in range(word_embed.shape[0])]
        print(cos_avg)
        ip_tokens = [np.inner(word_embed[i], word_embed[j]) for i in range(word_embed.shape[0]) for j in range(word_embed.shape[0]) if i != j]
        cos_tokens = [np.inner(word_embed[i], word_embed[j]) / (np.linalg.norm(word_embed[i]) * np.linalg.norm(word_embed[j])) for i in range(word_embed.shape[0]) for j in range(word_embed.shape[0]) if i != j]
        
        break

    df = pd.DataFrame({"cos_cls": cos_cls})
    # get the first element of every list
    df["cos_cls"] = df["cos_cls"].apply(lambda x: x[0])
    df["cos_cls"] = pd.cut(df["cos_cls"], bins=10)
    df = df.groupby("cos_cls").size().reset_index(name="count")
    df["cos_cls"] = df["cos_cls"].astype(str)
    chart1 = alt.Chart(df).mark_bar().encode(
        x=alt.X("cos_cls", title="Distance to [CLS] token"),
        y=alt.Y("count", title="Count"),
    ).properties(
        width=300,
        height=300,
        title="Distribution of distances to [CLS] token"
    )
    
    # plot the distribution of the distances to average token
    df = pd.DataFrame({"cos_avg": cos_avg})
    df["cos_avg"] = pd.cut(df["cos_avg"], bins=10)
    df = df.groupby("cos_avg").size().reset_index(name="count")
    df["cos_avg"] = df["cos_avg"].astype(str)
    chart2 = alt.Chart(df).mark_bar().encode(
        x=alt.X("cos_avg", title="Distance to average token"),
        y=alt.Y("count", title="Count"),
    ).properties(
        width=300,
        height=300,
        title="Distribution of distances to average token"
    )

    # plot the distribution of the distances between tokens
    df = pd.DataFrame({"cos_tokens": cos_tokens})
    df["cos_tokens"] = pd.cut(df["cos_tokens"], bins=10)
    df = df.groupby("cos_tokens").size().reset_index(name="count")
    df["cos_tokens"] = df["cos_tokens"].astype(str)
    chart3 = alt.Chart(df).mark_bar().encode(
        x=alt.X("cos_tokens", title="Distance between tokens"),
        y=alt.Y("count", title="Count"),
    ).properties(
        width=300,
        height=300,
        title="Distribution of distances between tokens"
    )

    chart = chart1 | chart2 | chart3
    chart.save(args.output_file)

        
    