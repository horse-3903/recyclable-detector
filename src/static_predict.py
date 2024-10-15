from ultralytics import YOLO
import os, shutil, stat
import random
import re
from tqdm import tqdm

print("Loading Model...", end="")
model = YOLO("./src/model.pt")
print("Done")
img = []

for label in os.listdir("./img"):
    img_dir = [f"./img/{label}/{dir}" for dir in os.listdir(f"./img/{label}")]
    random.shuffle(img_dir)

    img += img_dir[:20]

for dir in tqdm(iterable=img, desc="Predicting labels..."):
    model.predict(source=dir, save=True, conf=0.15, max_det=1)

    label = re.sub(r"[0-9]", "", os.listdir("./runs/detect/predict")[0][:-4])
    path = "./runs/detect/predict/" + os.listdir("./runs/detect/predict")[0]
    name = os.listdir("runs/detect/predict")[0]

    os.rename(path, f"./data/predict/{label}/{name}")

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)

shutil.rmtree("runs", onerror=on_rm_error)