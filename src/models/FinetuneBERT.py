from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import evaluate
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='./models/test/finetuned_bert/')

if __name__ == "__main__":
    args = parser.parse_args()
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    
    output_dir = args.output_dir
    training_args = TrainingArguments(
        output_dir=output_dir, 
        # per_device_train_batch_size=8,
        # per_device_eval_batch_size=8,
        # num_train_epochs=3,
        load_best_model_at_end=True,
        evaluation_strategy="no",
        save_strategy="no",
        # save the best model
    )
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    def tokenize_function(examples):
        return tokenizer(examples["content"], padding="max_length", truncation=True)
        #return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # dataset = load_dataset("tweet_eval", "sentiment")
    dataset = load_dataset("amazon_polarity")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Reduce size for time purposes
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))
    
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred): # metric is in main, will this work?
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    trainer.save_model()
    