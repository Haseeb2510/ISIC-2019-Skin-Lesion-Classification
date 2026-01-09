import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from data_cleaning.paths import PROCESSED_DATA, WORKED_IMGS, DATA

def main():
    df = pd.read_csv(os.path.join(PROCESSED_DATA, "ISIC_2019_Training_GroundTruth.csv"))
    df["filepath"] = df["image"].apply(lambda x: os.path.join(WORKED_IMGS, f"{x}.jpg"))

    # First split train vs temp
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3,
        stratify=df["label_idx"],
        random_state=42
    )

    # Then temp -> val and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label_idx"],
        random_state=42
    )


    def move_files(df: pd.DataFrame, split_name: str):
        for _, row in tqdm(df.iterrows(), total=len(df)):
            label = row["label_name"]
            src = row["filepath"]
            dst = f"{DATA}/{split_name}/{label}/"
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)
        
    move_files(train_df, "train")
    train_df = train_df.drop(columns=["filepath"])
    move_files(val_df, "val")
    val_df = val_df.drop(columns=["filepath"])
    move_files(test_df, "test")
    test_df = test_df.drop(columns=["filepath"])

    # Then save CSV
    train_df.to_csv(os.path.join(DATA, "train", "split_train.csv"), index=False)

    val_df.to_csv(os.path.join(f"{DATA}/val", "split_val.csv"), index=False)

    test_df.to_csv(os.path.join(f"{DATA}/test", "split_test.csv"), index=False)

    train_df["filepath"] = train_df.apply(
        lambda row: os.path.join(DATA, "train", row["label_name"], f"{row['image']}.jpg"), 
        axis=1
    )

    val_df["filepath"] = val_df.apply(
        lambda row: os.path.join(DATA, "val", row["label_name"], f"{row['image']}.jpg"),
        axis=1
    )

    test_df["filepath"] = test_df.apply(
        lambda row: os.path.join(DATA, "test", row["label_name"], f"{row['image']}.jpg"),
        axis=1
    )
    
    # Check if files exist
    print("="*60)
    print("CHECKING FILE EXISTENCE")
    print("="*60)
    
    for split_name, df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
        print(f"\n{split_name} SET:")
        print(f"  Total files in CSV: {len(df)}")
        
        # Check existence
        df["exists"] = df["filepath"].apply(os.path.exists)
        existing = df["exists"].sum()
        missing = len(df) - existing
        
        print(f"  Files found: {existing}")
        print(f"  Files missing: {missing}")
        
        if missing > 0:
            print(f"  Missing files (first 5):")
            missing_files = df[~df["exists"]][["image", "filepath"]].head()
            for _, row in missing_files.iterrows():
                print(f"    - {row['image']}: {row['filepath']}")
    
    print("\n" + "="*60)
    
    # Filter out missing files
    train_df = train_df[train_df["filepath"].apply(os.path.exists)].copy()
    val_df = val_df[val_df["filepath"].apply(os.path.exists)].copy()
    test_df = test_df[test_df["filepath"].apply(os.path.exists)].copy()
    
    print(f"\nAfter filtering missing files:")
    print(f"Train: {len(train_df)} files")
    print(f"Validation: {len(val_df)} files")
    print(f"Test: {len(test_df)} files")

    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/total_samples*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")

    print("\nClass distribution in each split:")
    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{split_name}:")
        class_counts = split_df["label_name"].value_counts()
        for label, count in class_counts.items():
            percentage = count / len(split_df) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()