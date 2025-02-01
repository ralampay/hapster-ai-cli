import os
import sys
import torch
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from yaspin import yaspin
from yaspin.spinners import Spinners
import time

from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.commons import load_huggingface_config, find_model_by_category

class Assist:
    def __init__(self, 
        settings=None, 
        models=None, 
        hugging_face_api_key=None,
        max_new_tokens=100,
        temperature=0.5,
        max_length=100,
        top_p=0.9,
        min_length=158,
        repetition_penalty=1.2
    ):
        self.settings = settings
        self.models = models
        self.hugging_face_api_key = hugging_face_api_key

        self.max_new_tokens=max_new_tokens
        self.temperature=temperature
        self.max_length=max_length
        self.top_p=top_p
        self.min_length=min_length
        self.repetition_penalty=repetition_penalty

        self.chat_model_id = self.settings["chat"]

        self.chat_model_config = find_model_by_category(self.models, "chat", self.chat_model_id)

        self.context = []

        print(colored(f"Initializing chat ai agent {self.chat_model_id}...", "magenta"))
        if self.chat_model_config["type"] == "huggingface":
            load_huggingface_config(self.chat_model_config, self.hugging_face_api_key)

        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id)
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            self.chat_model_id, 
            device_map="auto",   # Automatically map to GPU if available,
            offload_folder="./tmp"
        )

        self.pipe = pipeline(
            "text-generation", 
            model=self.chat_model,
            tokenizer=self.chat_tokenizer,
            max_new_tokens=self.max_new_tokens
        )

        self.model_kwargs = {
            #"max_length": max_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "top_p": self.top_p,                                     # Nucleus sampling, keeps only top tokens summing to 90% probability
            "pad_token_id": self.pipe.tokenizer.pad_token_id,    # Ensures padding does not affect the response
            "eos_token_id": self.pipe.tokenizer.eos_token_id,
            "truncation": True,
            "do_sample": True,
            "return_full_text": False
        }

        self.llm = HuggingFacePipeline(
            pipeline=self.pipe,
            model_kwargs=self.model_kwargs,
            pipeline_kwargs=self.model_kwargs
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant."),
            ("human", "{inquiry}")
        ])

        self.chain = self.prompt | self.llm
    
    def execute(self):
        while True:
            user_input = input(colored("Prompt: ", "light_cyan"))

            if user_input.lower() in ["exit", "quit"]:
                break

            start_time = time.time()
            with yaspin(text="", color="cyan") as spinner:
                response = self.generate_response(user_input)
                spinner.ok("✔")
                print(colored(response, "cyan"))

    def generate_response(self, user_input):
        response = self.chain.invoke({
            "inquiry": user_input
        })

        return response
