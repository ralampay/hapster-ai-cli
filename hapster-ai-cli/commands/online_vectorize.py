import openai
import os
import numpy as np
from langchain_openai import OpenAIEmbeddings
from utils.commons import get_file_extension, file_exists
from langchain_community.document_loaders import UnstructuredPowerPointLoader, TextLoader

class OnlineVectorize:
    def __init__(self, file_location="./tmp/test.txt", openai_api_key=""):
        self.file_location = file_location
        self.openai_api_key = openai_api_key
        self.file_extensions = {".txt", ".pptx"}

        self.ext = get_file_extension(self.file_location)

        self.loader = UnstructuredPowerPointLoader(self.file_location, mode="elements") if self.ext == ".pptx" else TextLoader(self.file_location)
        self.documents = self.loader.load()

        print("Initializing OpenAI embeddings...")
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

    def execute(self):
        for doc in self.documents:
            vector = self.embeddings.embed_query(doc.page_content)
            print(vector)
            print(f'Sum: {np.sum(vector)}')
