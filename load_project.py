import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.environ.get("API_KEY")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("recycling-detector").project("recyclables-and-garbage-detection")
dataset = project.version(3).download("yolov8")