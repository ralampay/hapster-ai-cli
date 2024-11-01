import os
import sys
import torch
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.commons import load_huggingface_config, find_model_by_category, extract_image_path

class IllustrateStory:
    def __init__(self, settings=None, models=None, hugging_face_api_key=None):
        self.settings = settings
        self.models = models
        self.hugging_face_api_key = hugging_face_api_key

        self.chat_model_id = self.settings["chat"]
        self.image_generator_model_id = self.settings["image_generator"]

        self.chat_model_config = find_model_by_category(self.models, "chat", self.chat_model_id)
        self.image_generator_model_config = find_model_by_category(self.models, "image_generator", self.image_generator_model_id)

        self.context = []

        if self.chat_model_config["type"] == "huggingface":
            load_huggingface_config(self.chat_model_config, self.hugging_face_api_key)

        if self.image_generator_model_config["type"] == "huggingface":
            load_huggingface_config(self.image_generator_model_config, self.hugging_face_api_key)

        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id)
        self.chat_model = AutoModelForCausalLM.from_pretrained(self.chat_model_id)

    def execute(self):
        self.print_meta()

        while True:
            prompt = input("Enter prompt (use @generate [filename].png to create an image or type exit / quit to end): ")

            if prompt.lower() in ["exit", "quit"]:
                break

            elif "@generate" in prompt:
                image_filepath = extract_image_path(prompt)

                image = self.generate_image(prompt)

                image.save(f"tmp/{image_filepath}")
                image.show()
            else:
                response = self.generate_chat_response(prompt)

                print("Chat AI:")
                print(response)

    def generate_image(self, prompt):
        self.context.append(f"User: {prompt}")
        prompt = "\n".join(self.context)

        scheduler = EulerDiscreteScheduler.from_pretrained(self.image_generator_model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(self.image_generator_model_id, scheduler=scheduler, torch_dtype=torch.float16)
        
        pipe.height = 128
        pipe.width = 128

        generated_image = pipe(prompt).images[0]

        return generated_image

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
        print(f"Image Generator Model: {self.settings.get("image_generator")}")
