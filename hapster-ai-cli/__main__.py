import argparse
import os
import sys
import yaml
import torch
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from commands.analyze_document import AnalyzeDocument
from commands.assist import Assist
from commands.illustrate_story import IllustrateStory
from commands.video_detect import VideoDetect
from commands.train_roboflow_ultralytics import TrainRoboflowUltralytics
from commands.generate_exam_item import GenerateExamItem
from commands.vectorize import Vectorize
from commands.online_vectorize import OnlineVectorize

# Load environment variables from the .env file
load_dotenv()

device = -1

# Determine device (0 for GPU, -1 for CPU)
if torch.cuda.is_available():
    device = 0  # Use the first GPU

def main():
    parser = argparse.ArgumentParser(description="HAC: Hapster AI CLI")

    #parser.add_argument("--command", type=str, default="analyze_document")
    #parser.add_argument("--command", type=str, default="generate_exam_item")
    #parser.add_argument("--command", type=str, default="vectorize")
    parser.add_argument("--command", type=str, default="online_vectorize")
    parser.add_argument("--video-file", type=str, default="video.mp4")
    parser.add_argument("--config-file", type=str, default="config.yaml")
    parser.add_argument("--file-location", type=str, default="./tmp/test.txt")

    args = parser.parse_args()

    command         = args.command
    video_file      = args.video_file
    config_file     = args.config_file
    file_location   = args.file_location

    if command == "assist":
        hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")
        models_loc = os.getenv("MODELS_LOC")
        mas_loc = os.getenv("MAS_LOC")

        # Open config files
        with open(models_loc, "r") as file:
            config_models = yaml.safe_load(file)

        with open(mas_loc, "r") as file:
            config_mas = yaml.safe_load(file)

        cmd = Assist(
            settings=config_mas.get('mas').get("assist"),
            models=config_models.get('models'),
            hugging_face_api_key=hugging_face_api_key
        )

        cmd.execute()

    elif command == "vectorize":
        cmd = Vectorize(
            file_location=file_location
        )

        cmd.execute()

    elif command == "online_vectorize":
        cmd = OnlineVectorize(
            file_location=file_location
        )

        cmd.execute()

    elif command == "generate_exam_item":
        openai_api_key = os.getenv("OPENAI_API_KEY")

        cmd = GenerateExamItem(
            openai_api_key=openai_api_key
        )

        cmd.execute()

    elif command == "analyze_document":
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

    elif command == "train_roboflow_ultralytics":
        roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

        if not config_file:
            config_file = 'default_ultralytics_config.yaml'

        # Open config files
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        cmd = TrainRoboflowUltralytics(
            api_key=roboflow_api_key,
            config=config
        )

        cmd.execute()

def print_settings():
    #print(f"HUGGING_FACE_API_KEY: {hugging_face_api_key}")
    print(f"MODELS_LOC: {models_loc}")
    print(f"MAS_LOC: {mas_loc}")
    print(f"DEVICE: {device}")

if __name__ == "__main__":
    #print_settings()

    main()
