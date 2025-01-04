import argparse
import os
import sys
import yaml
import torch
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from commands.analyze_document import AnalyzeDocument
from commands.illustrate_story import IllustrateStory
from commands.video_detect import VideoDetect

# Load environment variables from the .env file
load_dotenv()

device = -1

# Determine device (0 for GPU, -1 for CPU)
if torch.cuda.is_available():
    device = 0  # Use the first GPU

def main():
    parser = argparse.ArgumentParser(description="HAC: Hapster AI CLI")

    parser.add_argument("--command", type=str, default="analyze_document")
    parser.add_argument("--video-file", type=str, default="video.mp4")

    args = parser.parse_args()

    command     = args.command
    video_file  = args.video_file

    if command == "analyze_document":
        hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")
        models_loc = os.getenv("MODELS_LOC")
        mas_loc = os.getenv("MAS_LOC")

        # Open config files
        with open(models_loc, "r") as file:
            config_models = yaml.safe_load(file)

        with open(mas_loc, "r") as file:
            config_mas = yaml.safe_load(file)

        cmd = AnalyzeDocument(
            settings=config_mas.get('mas').get("document_analyzer"),
            models=config_models.get('models'),
            hugging_face_api_key=hugging_face_api_key
        )

        cmd.execute()

    elif command == "video_detect":
        model_file = os.getenv("DETECT_MODEL_FILE")

        cmd = VideoDetect(
            model_file=model_file,
            video_file=video_file
        )

        cmd.execute()

    print("Done. Have a nice day.")

def print_settings():
    #print(f"HUGGING_FACE_API_KEY: {hugging_face_api_key}")
    print(f"MODELS_LOC: {models_loc}")
    print(f"MAS_LOC: {mas_loc}")
    print(f"DEVICE: {device}")

if __name__ == "__main__":
    #print_settings()

    main()
