import os
import sys
import torch
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from yaspin import yaspin
from yaspin.spinners import Spinners
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.commons import load_huggingface_config, find_model_by_category

class Assist:
    def __init__(self, settings=None, models=None, hugging_face_api_key=None):
        self.settings = settings
        self.models = models
        self.hugging_face_api_key = hugging_face_api_key

        self.chat_model_id = self.settings["chat"]

        self.chat_model_config = find_model_by_category(self.models, "chat", self.chat_model_id)

        self.context = []

        print(colored(f"Initializing chat ai agent {self.chat_model_id}...", "magenta"))
        if self.chat_model_config["type"] == "huggingface":
            load_huggingface_config(self.chat_model_config, self.hugging_face_api_key)

        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id)
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            self.chat_model_id, 
            device_map="auto"   # Automatically map to GPU if available
        )
    
    def execute(self):
        while True:
            prompt = input(colored("Prompt: ", "light_cyan"))

            if prompt.lower() in ["exit", "quit"]:
                break

            start_time = time.time()
            with yaspin(text="", color="cyan") as spinner:
                response = self.generate_chat_response(prompt)
                spinner.ok("✔")
                #spinner.ok("")
                #spinner.ok(colored(f"AI: {response}", "cyan"))
                print(colored(f"AI: {response}", "cyan"))

    def generate_chat_response(self, prompt, max_length=100, max_new_tokens=50, temperature=0.9, top_p=0.9, min_length=158, repetition_penalty=1):
        user_chat = {
            "role": "User",
            "content": prompt
        }

        self.context.append(user_chat)

        generator = pipeline(
            "text-generation", 
            model=self.chat_model,
            tokenizer=self.chat_tokenizer
        )

        generation_kwargs = {
            #"max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,                                     # Nucleus sampling, keeps only top tokens summing to 90% probability
            "pad_token_id": generator.tokenizer.pad_token_id,    # Ensures padding does not affect the response
            "eos_token_id": generator.tokenizer.eos_token_id,
            "truncation": True,
            "do_sample": True
        }

        input_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in self.context]).join(f"\n\n\n{prompt}")

        generated_text = generator(
            prompt,
            **generation_kwargs
        )

        response = generated_text[0]["generated_text"]

        ai_chat = {
            'role': 'assistant',
            'content': response
        }

        self.context.append(ai_chat)

        return response
