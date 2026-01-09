import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from data_cleaning.paths import PROCESSED_DATA, PROCESSED_IMGS

df = pd.read_csv(os.path.join(PROCESSED_DATA, "ISIC_2019_Training_GroundTruth.csv"))

# Verify all resized images 
df["image_path"] = df["image"].apply(lambda x: os.path.join(PROCESSED_IMGS, f"{x}.jpg"))

df["exists"] = df["image_path"].apply(os.path.exists)

missing = df[df["exists"] == False]
print("Missing images:", len(missing))
missing.head()

# Check for corrupted images
corrupted = []

for p in tqdm(df["image_path"]):
    try:
        img = Image.open(p)
        img.verify()        # verify integrity
    except:
        corrupted.append(p)

print("Corrupted images: ", corrupted)

# If images corrupted
df = df[~df["image_path"].isin(corrupted)]
df = df[df["exists"] == True]
df = df.reset_index(drop=True)


# Class distribution
class_counts = df["label_name"].value_counts()

print(class_counts)
class_counts.plot(kind="bar", title="Class Distribution", figsize=(10,5))
plt.show()

# Sample images per class
classes = df["label_name"].unique()

for c in classes:
    samples = df[df["label_name"] == c].sample(5)["image_path"].values

    plt.figure(figsize=(8,3))
    plt.suptitle(f"Class: {c}", fontsize=14)

    for i, path in enumerate(samples):
        img = Image.open(path)
        plt.subplot(1,5,i+1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

# Collect image properties
shapes = []

for p in tqdm(df["image_path"]):
    img = Image.open(p)
    shapes.append(img.size)

shapes = np.array(shapes)

print("Min size: ", shapes.min(axis=0))
print("Max size: ", shapes.max(axis=0))
print("Mean size: ", shapes.mean(axis=0))