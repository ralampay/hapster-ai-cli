import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.commons import load_huggingface_config

class AnalyzeDocument:
    def __init__(self, settings=None, models=None, hugging_face_api_key=None):
        self.settings = settings
        self.models = models
        self.hugging_face_api_key = hugging_face_api_key

        self.chat_model_config = next((item for item in self.models.get("chat") if item.get("model_id") == self.settings["chat"]), None)

        self.context = []

        if self.chat_model_config["type"] == "huggingface":
            load_huggingface_config(self.chat_model_config, self.hugging_face_api_key)

        self.model_id = self.chat_model_config["model_id"]

        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.chat_model = AutoModelForCausalLM.from_pretrained(self.model_id)

    def execute(self):
        self.print_meta()

        while True:
            prompt = input("Enter prompt (use @document [filepath] to start summarizing or type exit / quit to end): ")

            if prompt.lower() in ["exit", "quit"]:
                break

            response = self.generate_chat_response(prompt)

            print("AI:")
            print(response)

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
