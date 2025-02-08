from transformers import AutoTokenizer, AutoConfig, AutoModel
from termcolor import colored
from yaspin import yaspin
from yaspin.spinners import Spinners
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

class Vectorize:
    def __init__(self,
        model="deepseek-ai/deepseek-llm-7b-base",
        text="Hello world!",
        device="cpu"
    ):
        self.model = model
        self.text = text
        self.device = device

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )

        self.model_kwargs = {
            'device': self.device
        }

        self.encode_kwargs = {
            'normalize_embeddings': False
        }

        print(colored(f"Initializing model {self.model}...", "magenta"))

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

    def execute(self):
        # Split the long text
        chunks = self.text_splitter.split_text(self.text)

        for i, chunk in enumerate(chunks):
            print(f"{i}: {chunk}")

            query_result = self.embeddings.embed_query(self.text)

            print(len(query_result))
