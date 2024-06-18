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


def train_gpt2_model(train_data, eval_data):
    """Train the GPT2 model by passing in the training and evaluation phishing email data"""

    # Only the 'cleaned_message' column is required
    train_data = Dataset.from_pandas(train_data[["cleaned_message"]])
    eval_data = Dataset.from_pandas(eval_data[["cleaned_message"]])

    # Initialise the model
    print("Initialising GPT2 Model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("GPT2 model initialised.")

    # Tokenize the train dataset
    print("Tokenizing the train dataset...")
    tokenized_train_dataset = train_data.map(tokenize_function, batched=True, remove_columns=["cleaned_message"])
    print("Train dataset tokenized.")

    # Tokenize the evaluation dataset
    print("Tokenizing the evaluation dataset...")
    tokenized_eval_dataset = eval_data.map(tokenize_function, batched=True, remove_columns=["cleaned_message"])
    print("Evaluation dataset tokenized.")

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
        fp16=False,
        logging_dir="./logs",
        logging_steps=5,
        eval_steps=10, # Evaluate every 10 steps
        eval_strategy="steps",
        use_cpu=True # Enable CPU training
    )
    print("Training arguments set.")

    # Initialise the trainer
    print("Initialising the trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer
    )
    print("Trainer initialised.")

    # Train the model
    print("Training the GPT2 model...")
    trainer.train()
    print("Training completed successfully.")

    # Safe the final model and tokenizer
    print("Saving the final model and tokenizer...")
    trainer.save_model("gpt2-model-final")
    tokenizer.save_pretrained("gpt2-model-final")
    print("Model and tokenizer saved successfully.")

