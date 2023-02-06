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
parser.add_argument('--output_file', default='./visualization/token-sim-heatmap-sbert.html')
parser.add_argument('--model', default='bert-base-cased')
parser.add_argument('--tokenizer', default='bert-base-cased')

if __name__ == "__main__":
    args = parser.parse_args()
    
    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else -1
    
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) 
    # model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') 
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

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

    # get a matrix with the cosine similarity of all tokens
    cos = np.inner(embed, embed) / (np.linalg.norm(embed, axis=1)[:, None] * np.linalg.norm(embed, axis=1)[None, :])
    cos = np.round(cos, 3)

    # plot an altair heatmap using the values in cos (so that color is the value fo the token pair in cos) and the token names in tokens
    df = pd.DataFrame(cos, columns=tokens, index=tokens)
    # turn grid into columnar data
    df = df.stack().reset_index()
    df.columns = ['index', 'columns', 'value']

    # sort x by tokens[::-1] and y by tokens, axis title equals x and y, no legend, title is the sentence
    heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X('index:N', sort=tokens, axis=alt.Axis(title='')),
        y=alt.Y('columns:N', sort=tokens, axis=alt.Axis(title='')),
        color=alt.Color('value:Q', legend=None),
        tooltip=['index', 'columns', 'value']
    ).properties(
        width=600,
        height=600,
        title=f"SBERT - {dataset['text'][i]}"
    )
    heatmap.save(args.output_file)