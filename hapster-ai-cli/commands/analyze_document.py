import os
import sys
import torch
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from yaspin import yaspin
from yaspin.spinners import Spinners

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.commons import load_huggingface_config, find_model_by_category, extract_text_from_pdf, extract_file_path, extract_text_from_txt, get_file_extension

class AnalyzeDocument:
    def __init__(self, settings=None, models=None, hugging_face_api_key=None):
        self.settings = settings
        self.models = models
        self.hugging_face_api_key = hugging_face_api_key

        self.chat_model_id = self.settings["chat"]
        self.text_summarization_model_id = self.settings["text_summarization"]

        self.chat_model_config = find_model_by_category(self.models, "chat", self.chat_model_id)
        self.text_summarization_model_config = find_model_by_category(self.models, "text_summarization", self.text_summarization_model_id)

        self.context = []

        print(colored(f"Initializing chat ai agent {self.chat_model_id}...", "magenta"))
        if self.chat_model_config["type"] == "huggingface":
            load_huggingface_config(self.chat_model_config, self.hugging_face_api_key)

        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id)
        self.chat_model = AutoModelForCausalLM.from_pretrained(self.chat_model_id)

        print(colored(f"Initializing text summarizer ai agent {self.text_summarization_model_id}...", "magenta"))
        if self.text_summarization_model_config["type"] == "huggingface":
            load_huggingface_config(self.text_summarization_model_config, self.hugging_face_api_key)

        self.text_summarization_tokenizer = AutoTokenizer.from_pretrained(self.text_summarization_model_id)
        self.text_summarization_model = AutoModelForSeq2SeqLM.from_pretrained(self.text_summarization_model_id)

    def execute(self):
        while True:
            prompt = input(colored("Enter prompt (use @summarize [filepath] to start summarizing or type exit / quit to end):\n", "light_cyan"))

            if prompt.lower() in ["exit", "quit"]:
                break

            elif "@summarize" in prompt:
                doc_filepath = extract_file_path(prompt)

                if doc_filepath is not None:

                    file_ext = get_file_extension(doc_filepath)

                    if file_ext == ".pdf":
                        content = extract_text_from_pdf(doc_filepath)
                    else:
                        content = extract_text_from_txt(doc_filepath)

                    prompt = f"Summarize --- {content} ---"

                    with yaspin(text="Generating response...", color="cyan") as spinner:
                        response = self.generate_document_summary(prompt)
                        spinner.ok("✔")

                    print(colored("Text Summarizer AI:\n", "light_blue"))
                    print(response)
                else:
                    print("File not found.")
            else:
                with yaspin(text="Generating response...", color="cyan") as spinner:
                    response = self.generate_chat_response(prompt)
                    spinner.ok("✔")

                print(colored("Chat AI:\n", "light_blue"))
                print(response)

            print("\n")

    def generate_document_summary(self, prompt, max_length=32000, max_new_tokens=150, temperature=0.7, top_p=0.9):
        summarizer = pipeline("summarization", model=self.text_summarization_model_id)

        result = summarizer(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )

        response = result[0]["summary_text"]

        self.context.append(f"AI: {response}")

        return response

    def generate_chat_response(self, prompt, max_length=10000, max_new_tokens=150, temperature=0.5, top_p=0.9, min_length=158):
        self.context.append(f"User: {prompt}")

        prompt = "\n".join(self.context)

        generator = pipeline(
            "text-generation", 
            model=self.chat_model_id
        )

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,                   # Maximum length of the generated response
            "temperature": temperature,                         # Balances randomness and determinism
            "top_k": 50,                                        # Filters the top 50 tokens to consider for each step
            "top_p": top_p,                                     # Nucleus sampling, keeps only top tokens summing to 90% probability
            "do_sample": True,
            "pad_token_id": generator.tokenizer.eos_token_id    # Ensures padding does not affect the response
        }

        generated_text = generator(
            prompt,
            **generation_kwargs
        )

        response = generated_text[0]["generated_text"]

        self.context.append(f"AI: {response}")

        return response
