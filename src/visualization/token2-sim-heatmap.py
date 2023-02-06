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
parser.add_argument('--output_file', default='./visualization/_sbert.html')
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
    
    dataset = load_dataset("stsb_multi_mt", name="en", split="dev")
    
    # get a sentence with high similarity score, which can be 0 to 5
    dataset = dataset.filter(lambda example: example["similarity_score"] == 4)

    i = 1
    tokenized_sentence = tokenizer(dataset["sentence1"][i], return_tensors="pt")
    _tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence["input_ids"][0])
    _embed = model(**tokenized_sentence)[0].detach().squeeze(0).numpy()

    embed_avg = np.expand_dims(np.average(_embed, axis=0), axis=0)
    embed_max = np.expand_dims(np.max(_embed, axis=0), axis=0)
    embed1 = np.concatenate((_embed, embed_avg, embed_max), axis=0)
    tokens1 = _tokens + ["[AVG]", "[MAX]"]

    tokenized_sentence = tokenizer(dataset["sentence2"][i], return_tensors="pt")
    _tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence["input_ids"][0])
    _embed = model(**tokenized_sentence)[0].detach().squeeze(0).numpy()
    
    embed_avg = np.expand_dims(np.average(_embed, axis=0), axis=0)
    embed_max = np.expand_dims(np.max(_embed, axis=0), axis=0)
    embed2 = np.concatenate((_embed, embed_avg, embed_max), axis=0)
    tokens2 = _tokens + ["[AVG]", "[MAX]"]
    
    # get a matrix with the cosine similarity of all tokens
    cos = np.inner(embed1, embed2) / (np.linalg.norm(embed1, axis=1)[:, None] * np.linalg.norm(embed2, axis=1)[None, :])
    cos = np.round(cos, 3)

    # plot an altair heatmap using the values in cos (so that color is the value of the token pair in cos) and the token names in tokens
    # ob the y axis put the tokens from sentence 1, and on the x axis put the tokens from sentence 2
    df = pd.DataFrame(cos, index=tokens1, columns=tokens2)
    # turn grid into columnar data 
    df = df.stack().reset_index()
    df.columns = ['index', 'index2', 'value']
    heatmap = alt.Chart(df.reset_index()).mark_rect().encode(
        x=alt.X('index:O', sort=tokens1, axis=alt.Axis(title='Sentence 2')),
        y=alt.Y('index2:O', sort=tokens2, axis=alt.Axis(title='Sentence 1')),
        color=alt.Color('value:Q', legend=None),
        tooltip=['index', 'index2', 'value']
    ).properties(
        width=600,
        height=600,
        title="SBERT"
    )
    
    heatmap.save(args.output_file)