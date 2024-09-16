import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables
load_dotenv()

# Authenticate with Hugging Face
from huggingface_hub import login

api_token = os.getenv('HUGGINGFACE_API_TOKEN')
if api_token:
    login(token=api_token)
else:
    raise ValueError("HUGGINGFACE_API_TOKEN environment variable not set")

# Model ID for Meta-Llama-3-70B-Instruct
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Define pipeline for text generation
def generate_text(messages, max_new_tokens=256, temperature=0.6, top_p=0.9):
    inputs = tokenizer(messages, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
messages = [
    "You are a pirate chatbot who always responds in pirate speak!",
    "Who are you?"
]

# Generate response
response = generate_text(messages)
print(response)
