import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model_and_tokenizer(model_dir="gpt2-model-final"):
    """Load the trained GPT2 model and tokenizer."""
    print("Loading the trained model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def generate_text(prompt, max_length=100, num_return_sequences=1):
    """Generate text using the trained GPT2 model and tokenizer."""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Generating synthetic text based on given prompt
    print(f"Generating text with prompt: '{prompt}'...")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    generated_texts=[tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    print("Text generation completed successfully.")
    return generated_texts

