import os

from huggingface_hub import hf_hub_download
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

print(f'HUGGING_FACE_API_KEY: {HUGGING_FACE_API_KEY}')

# Determine device (0 for GPU, -1 for CPU)
if torch.cuda.is_available():
    device = 0  # Use the first GPU
else:
    device = -1  #

# https://huggingface.co/lmsys/fastchat-t5-3b-v1.0
model_id = "ericzzz/falcon-rw-1b-instruct-openorca"

filenames = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"
]

for filename in filenames:
    downloaded_model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        token=HUGGING_FACE_API_KEY
    )

    print(f'Downloaded file: {downloaded_model_path}')

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = pipeline(
   'text-generation',
   model=model_id,
   tokenizer=tokenizer,
   torch_dtype=torch.bfloat16,
   device_map='auto',
)

prompt = '''
Autoreply to an angry employee
'''

response = pipeline(
   prompt, 
   max_length=1000,
   repetition_penalty=1.05,
   truncation=True
)

print(response[0]['generated_text'])
