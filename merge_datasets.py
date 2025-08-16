import pandas as pd
import os

# Optional: Make sure the save path exists (in this case, 'dataset')
os.makedirs("dataset", exist_ok=True)

# Load datasets from 'dataset' folder
jigsaw_df = pd.read_csv(r"C:\Projects\keyboard1\dataset\jigsaw.csv")
olid_df = pd.read_csv(r"C:\Projects\keyboard1\dataset\olid.csv")

# Merge and shuffle
combined_df = pd.concat([jigsaw_df, olid_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged dataset to 'dataset' folder
combined_df.to_csv("dataset/combined_dataset.csv", index=False)

print("✅ Combined dataset saved to: dataset/combined_dataset.csv")
print("Total records:", len(combined_df))
