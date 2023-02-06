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
parser.add_argument('--output_file', default='./visualization/embed-unit-norm.html')
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
    
    dataset = dataset.shuffle(seed=14).select(range(100))

    i = 35
    tokenized_sentence = tokenizer(dataset["text"][i], return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence["input_ids"][0])
    embed = model(**tokenized_sentence)[0].detach().squeeze(0).numpy()

    embed_avg = np.expand_dims(np.average(embed, axis=0), axis=0)
    embed_max = np.expand_dims(np.max(embed, axis=0), axis=0)
    embed = np.concatenate((embed, embed_avg, embed_max), axis=0)
    tokens = tokens + ["[AVG]", "[MAX]"]

    # get the norm of each of the embeddings
    norm = np.linalg.norm(embed, axis=1)
    
    # plot an altair barchart of the norms of the embeddings, so that the x axis is the token and the y axis is the norm
    # sort the barchart on the X axis by the norm decreasingly
    df = pd.DataFrame(norm, columns=["norm"], index=tokens)
    df = df.stack().reset_index()
    df.columns = ['index', 'columns', 'value']
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('index', sort=tokens, axis=alt.Axis(title='Token')),
        y=alt.Y('value', axis=alt.Axis(title='Norm of the embedding')),
    ).properties(
        width=800,
        height=400,
        title="Norm of the BERT embeddings"
    )
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') 
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    tokenized_sentence = tokenizer(dataset["text"][i], return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence["input_ids"][0])
    embed = model(**tokenized_sentence)[0].detach().squeeze(0).numpy()
    embed_avg = np.expand_dims(np.average(embed, axis=0), axis=0)
    embed_max = np.expand_dims(np.max(embed, axis=0), axis=0)
    embed = np.concatenate((embed, embed_avg, embed_max), axis=0)
    tokens = tokens + ["[AVG]", "[MAX]"]

    # get the norm of each of the embeddings
    norm = np.linalg.norm(embed, axis=1)
    
    # plot an altair barchart of the norms of the embeddings, so that the x axis is the token and the y axis is the norm
    # sort the barchart on the X axis by the norm decreasingly
    df = pd.DataFrame(norm, columns=["norm"], index=tokens)
    df = df.stack().reset_index()
    df.columns = ['index', 'columns', 'value']
    chart2 = alt.Chart(df).mark_bar().encode(
        x=alt.X('index', sort=tokens, axis=alt.Axis(title='Token')),
        y=alt.Y('value', axis=alt.Axis(title='Norm of the embedding')),
    ).properties(
        width=800,
        height=400,
        title="Norm of the SBERT embeddings"
    )

    (chart | chart2).save(args.output_file)
    