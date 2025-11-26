import os
import pandas as pd

# Base dataset folder
base_dir = r"D:\BICS\INSIGHT\data\slit_lamp"

# Folders for cataract vs normal
categories = {
    "1_normal": 0,   # label 0
    "2_cataract": 1  # label 1
}

data = []

for folder, label in categories.items():
    folder_path = os.path.join(base_dir, folder)
    
    if not os.path.exists(folder_path):
        print(f" Warning: folder not found -> {folder_path}")
        continue
    
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(folder_path, file)
            data.append([filepath, label])

# Save to CSV
df = pd.DataFrame(data, columns=["filepath", "label"])
csv_path = os.path.join(base_dir, "labels_slitlamp.csv")
df.to_csv(csv_path, index=False)

print(f" Labels CSV created: {csv_path}")
print(df.head())
