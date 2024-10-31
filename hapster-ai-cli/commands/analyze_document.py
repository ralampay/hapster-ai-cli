import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.commons import load_huggingface_config, find_model_by_category, extract_text_from_pdf, extract_file_path

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

        if self.chat_model_config["type"] == "huggingface":
            load_huggingface_config(self.chat_model_config, self.hugging_face_api_key)

        if self.text_summarization_model_config["type"] == "huggingface":
            load_huggingface_config(self.text_summarization_model_config, self.hugging_face_api_key)

        self.chat_model_id = self.chat_model_config["model_id"]
        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id)
        self.chat_model = AutoModelForCausalLM.from_pretrained(self.chat_model_id)

        self.text_summarization_model_id = self.text_summarization_model_config["model_id"]
        self.text_summarization_tokenizer = AutoTokenizer.from_pretrained(self.text_summarization_model_id)
        self.text_summarization_model = AutoModelForSeq2SeqLM.from_pretrained(self.text_summarization_model_id)

    def execute(self):
        self.print_meta()

        while True:
            prompt = input("Enter prompt (use @document [filepath] to start summarizing or type exit / quit to end): ")

            print(f"prompt: {prompt}")

            if prompt.lower() in ["exit", "quit"]:
                break

            elif "@document" in prompt:
                doc_filepath = extract_file_path(prompt)

                if doc_filepath is not None:
                    content = extract_text_from_pdf(doc_filepath)

                    prompt = f"Summarize --- {content} ---"

                    response = self.generate_document_summary(prompt)

                    print("Text Summarizer AI:")
                    print(response)
                else:
                    print("File not found.")
            else:
                response = self.generate_chat_response(prompt)

                print("Chat AI:")
                print(response)

    def generate_document_summary(self, prompt, max_length=20240, max_new_tokens=150, temperature=0.7, top_p=0.9):
        prompt = "\n".join(self.context)

        inputs = self.text_summarization_tokenizer(prompt, padding="max_length", max_length=max_length, return_tensors="pt", truncation=True)

        global_attention_mask = torch.zeros_like(inputs["attention_mask"])

        outputs = self.text_summarization_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            global_attention_mask=global_attention_mask
        )

        response = ""

        for output in outputs:
            response += self.text_summarization_tokenizer.decode(output, skip_special_tokens=True)

        self.context.append(f"AI: {response}")

        return response

    def generate_chat_response(self, prompt, max_length=10000, max_new_tokens=150, temperature=0.7, top_p=0.9):
        self.context.append(f"User: {prompt}")
        prompt = "\n".join(self.context)

        inputs = self.chat_tokenizer(prompt, return_tensors="pt")

        outputs = self.chat_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.chat_tokenizer.eos_token_id,
            attention_mask=inputs["attention_mask"]
        )

        response = ""

        for output in outputs:
            response += self.chat_tokenizer.decode(output, skip_special_tokens=True)

        response = response[len(prompt):].strip()

        self.context.append(f"AI: {response}")

        return response

    def print_meta(self):
        print("AI Operation: Analyize Document")
        print(f"Chat Model: {self.settings.get("chat")}")
        print(f"Text Summarization Model: {self.settings.get("text_summarization")}")
