import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from tqdm import tqdm
import pandas as pd
import cv2, numpy as np
from data_cleaning.paths import PROCESSED_DATA, PROCESSED_IMGS, WORKED_IMGS

# CLAHE - improve local contrast
def apply_clahe_rgb(img):
    """Apply CLAHE to RGB image - expects numpy array in RGB format"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    return cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)

# Hair removal - detect dark thin lines and inpaint
def remove_hairs(img_rgb):
    """Remove hairs from RGB image - expects numpy array in RGB format"""
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Black-hat to find hair-like structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    # Threshold the blackhat
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Dilate mask a bit
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    # Inpaint the RGB image
    inpainted = cv2.inpaint(img_rgb, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def main():
    df = pd.read_csv(os.path.join(PROCESSED_DATA, "ISIC_2019_Training_GroundTruth.csv"))
    for img_name in tqdm(df["image"]):
        src_path = os.path.join(PROCESSED_IMGS, f"{img_name}.jpg")
        dst_path = os.path.join(WORKED_IMGS, f"{img_name}.jpg")
        # Skip if already resized
        if os.path.exists(dst_path):
            continue
        try:
            # Read with OpenCV
            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                print(f"Could not read image: {src_path}")
                continue
                
            # Convert BGR to RGB for processing
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply processing
            img_processed = apply_clahe_rgb(img_rgb)
            img_processed = remove_hairs(img_processed)
            
            # Convert back to BGR for saving with OpenCV
            img_processed_bgr = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, img_processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        except Exception as e:
            print(f"Error processing: {img_name}", e)
        

main()
