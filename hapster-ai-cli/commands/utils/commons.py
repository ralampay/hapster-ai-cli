from huggingface_hub import hf_hub_download
import PyPDF2
from markdown import markdown
import re
import os

def get_file_extension(path):
    return os.path.splitext(path)[1]

def extract_image_path(text):
    match = re.search(r"@generate\s+([^\s]+)", text)
    if match:
        return match.group(1)
    return None

def extract_file_path(text):
    match = re.search(r"@document\s+([^\s]+)", text)
    if match:
        return match.group(1)
    return None

def find_model_by_category(models_data, category, model_id):
    for model in models_data:
        if category in model.get('category', []) and model.get('model_id') == model_id:
            return model

    return None

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        content = file.read()

        return content

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        text = ""

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        return markdown(text)

def load_huggingface_config(settings, hugging_face_api_key):
    print("Loading files...")

    for filename in settings["filenames"]:
        downloaded_model_path = hf_hub_download(
            repo_id=settings["model_id"],
            filename=filename,
            token=hugging_face_api_key
        )

        print(f"Downloaded file: {downloaded_model_path}")
