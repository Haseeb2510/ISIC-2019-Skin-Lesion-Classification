import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from PIL import Image
from tqdm import tqdm
from data_cleaning.paths import DATA, RAW_DATA, RAW_IMGS, PROCESSED_DATA, PROCESSED_IMGS

df = pd.read_csv(os.path.join(RAW_DATA, "ISIC_2019_Training_GroundTruth.csv"))

# Class columns in order
class_cols = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]

# Determine label column for one-hot encoding
df["label_name"] = df[class_cols].idxmax(axis=1)

# Map label_name -> integer class
class_to_idx = {c:i for i,c in enumerate(class_cols)}
df["label_idx"] = df["label_name"].map(class_to_idx)

# Resize Images to reduce dataset size
target_size = (512,512)                 # or 244x244 for fast training

bad_images = []
processed_imgs = os.path.join(PROCESSED_IMGS)

for img_name in tqdm(df["image"]):
    src_path = os.path.join(RAW_IMGS, f"{img_name}.jpg")
    dst_path = os.path.join(processed_imgs, f"{img_name}.jpg")
    # Skip if already resized
    if os.path.exists(dst_path):
        continue

    try:
        img = Image.open(src_path)
        img = img.convert("RGB")    # ensure 3 channels
        img = img.resize(target_size, Image.BILINEAR) # type: ignore
        img.save(dst_path, "JPEG", quality=90)
    except Exception as e:
        bad_images.append((img_name, str(e)))

print("Corrupted Images: ", bad_images)

df.to_csv(os.path.join(PROCESSED_DATA, "ISIC_2019_Training_GroundTruth.csv"), index=False)