from huggingface_hub import hf_hub_download

def load_huggingface_config(settings, hugging_face_api_key):
    print("Loading files...")

    for filename in settings["filenames"]:
        downloaded_model_path = hf_hub_download(
            repo_id=settings["model_id"],
            filename=filename,
            token=hugging_face_api_key
        )

        print(f"Downloaded file: {downloaded_model_path}")
