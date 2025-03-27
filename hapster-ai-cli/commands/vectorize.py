from transformers import AutoTokenizer, AutoConfig, AutoModel
from termcolor import colored
from yaspin import yaspin
from yaspin.spinners import Spinners
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPowerPointLoader, TextLoader
import numpy as np
from utils.commons import get_file_extension, file_exists

class Vectorize:
    def __init__(self,
        model="deepseek-ai/deepseek-llm-7b-base",
        chunk_size=100,
        chunk_overlap=20,
        file_location="./tmp/test.txt",
        device="cpu"
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_location = file_location
        self.device = device

        self.file_extensions = {
            ".txt",
            ".pptx"
        }

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

        print(colored(f"Checking file {self.file_location}...", "magenta"))
        self.extension = get_file_extension(self.file_location)

        if self.extension not in self.file_extensions:
            print(colored(f"Invalid extension {extension} for {self.file_location}", "red"))
            exit(1)

        if not file_exists(self.file_location):
            print(colored(f"File {self.file_location} does not exist", "red"))
            exit(1)

        if self.extension == ".pptx":
            print(colored(f"Loading documents using UnstructuredPowerPointLoader for {self.file_location}...", "magenta"))
            self.loader = UnstructuredPowerPointLoader(self.file_location, mode="elements")
        elif self.extension == ".txt":
            print(colored(f"Loading documents using TextLoader for {self.file_location}...", "magenta"))
            self.loader = TextLoader(self.file_location)
        else:
            print(colored(f"No loader supported for extension {self.extension}", "red"))
            exit(1)

        self.documents = self.loader.load()

        print(colored(f"Initializing model {self.model}...", "magenta"))

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

    def execute(self):
        for doc in self.documents:
            content = doc.page_content
            vector = self.embeddings.embed_query(content)

            print(vector)
