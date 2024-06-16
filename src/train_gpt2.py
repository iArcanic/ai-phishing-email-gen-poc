import pandas as pd
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from data_preprocessing import load_data

# Initialise the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    tokenized = tokenizer(
        examples['cleaned_message'],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


def train_gpt2_model(train_data):
    """Train the GPT2 model by passing in the training phishing email data"""

    # Only the 'cleaned_message' column is required
    train_data = Dataset.from_pandas(train_data[["cleaned_message"]])

    # Initialise the model
    print("Initialising GPT2 Model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("GPT2 model initialised.")

    # Tokenize the dataset
    tokenized_datasets = train_data.map(tokenize_function, batched=True, remove_columns=["cleaned_message"])

    # Initialise the training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="gpt2-model-results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        save_steps=10,
        save_total_limit=2,
        fp16=True,
        evaluation_strategy="steps"
    )
    print("Training arguments set.")

    # Initialise the trainer
    print("Initialising the trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer
    )
    print("Trainer initialised.")

    # Train the model
    print("Training the GPT2 model...")
    trainer.train()
    print("Training completed successfully.")
