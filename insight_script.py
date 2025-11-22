# insight_script.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import numpy as np

# --- CONFIG ---
# Paths are correct from our last fix
DATA_DIR = "D:/BICS/insightweb/data/images/preprocessed_images/" 
LABEL_FILE = "D:/BICS/insightweb/data/images/full_df.csv"
# ---------------

def load_data(label_file):
    """
    Loads the CSV and creates an ACCURATE, IMAGE-LEVEL binary label
    by checking the specific diagnostic keywords for each eye.
    """
    df = pd.read_csv(label_file)
    
    print("Parsing image-level labels...")
    
    # 1. Process left eye images
    left_df = df[['Left-Fundus', 'Left-Diagnostic Keywords']].copy()
    left_df.rename(columns={'Left-Fundus': 'filename'}, inplace=True)
    # Create binary label: 1 if 'cataract' (lowercase) is in the keyword string
    left_df['Cataract'] = left_df['Left-Diagnostic Keywords'].apply(lambda x: 1 if 'cataract' in str(x) else 0)
    
    # 2. Process right eye images
    right_df = df[['Right-Fundus', 'Right-Diagnostic Keywords']].copy()
    right_df.rename(columns={'Right-Fundus': 'filename'}, inplace=True)
    # Create binary label: 1 if 'cataract' (lowercase) is in the keyword string
    right_df['Cataract'] = right_df['Right-Diagnostic Keywords'].apply(lambda x: 1 if 'cataract' in str(x) else 0)
    
    # 3. Combine them
    # We only want the filename and the final label
    all_images_df = pd.concat([
        left_df[['filename', 'Cataract']], 
        right_df[['filename', 'Cataract']]
    ]).dropna()
    
    # 4. Clean known bad file (a known issue in the ODIR dataset)
    all_images_df = all_images_df[all_images_df['filename'] != 'db_no_fovea.jpg'].reset_index(drop=True)
    
    # Convert 'Cataract' to integer
    all_images_df['Cataract'] = all_images_df['Cataract'].astype(int)
    
    print("Label parsing complete.")
    return all_images_df

def plot_class_balance(df):
    if df.empty:
        print("Skipping class balance plot due to data loading error.")
        return
        
    print("Plotting class balance...")
    plt.figure(figsize=(8, 6))
    
    if 'Cataract' not in df.columns:
        print("Error: 'Cataract' column not found for plotting.")
        return
        
    counts = df['Cataract'].value_counts().sort_index()
    
    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)
    total = count_0 + count_1
    
    if total == 0:
        print("No data to plot for class balance.")
        return

    sns.countplot(x='Cataract', data=df)
    plt.title('Class Balance (0 = Not Cataract, 1 = Cataract)')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    
    plt.xticks(ticks=[0, 1], labels=[f"0: Not Cataract\n({count_0} | {count_0/total:.1%})", 
                                      f"1: Cataract\n({count_1} | {count_1/total:.1%})"])
    plt.savefig("class_balance.png")
    print("Saved class_balance.png")
    plt.show()

def show_example_images(df, data_dir):
    if df.empty:
        print("Skipping example images due to data loading error.")
        return
    
    print("Showing example images...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    try:
        # Get 4 examples for each class
        cataract_samples = df[df['Cataract'] == 1].sample(4)['filename']
        non_cataract_samples = df[df['Cataract'] == 0].sample(4)['filename']
    except (ValueError, KeyError) as e:
        print(f"Error sampling images: {e}")
        print("Check class balance plot. Do you have at least 4 images for each class?")
        return

    # Plot Cataract Examples
    for i, img_name in enumerate(cataract_samples):
        img_path = os.path.join(data_dir, img_name)
        try:
            img = Image.open(img_path)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Class: 1 (Cataract)\n{img_name}")
            axes[0, i].axis('off')
        except FileNotFoundError:
            print(f"Warning: Could not find image {img_path}")
            axes[0, i].set_title(f"Image Not Found\n{img_name}")
            axes[0, i].axis('off')
        
    # Plot Non-Cataract Examples
    for i, img_name in enumerate(non_cataract_samples):
        img_path = os.path.join(data_dir, img_name)
        try:
            img = Image.open(img_path)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Class: 0 (Not Cataract)\n{img_name}")
            axes[1, i].axis('off')
        except FileNotFoundError:
            print(f"Warning: Could not find image {img_path}")
            axes[1, i].set_title(f"Image Not Found\n{img_name}")
            axes[1, i].axis('off')
        
    plt.suptitle("Example Images from Dataset", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("example_images.png")
    print("Saved example_images.png")
    plt.show()

if __name__ == "__main__":
    main_df = load_data(LABEL_FILE)
    
    if not main_df.empty:
        print(main_df.head())
        print(f"\nTotal images to analyze: {len(main_df)}")
        print("\nClass counts:")
        print(main_df['Cataract'].value_counts())
        
        plot_class_balance(main_df)
        show_example_images(main_df, DATA_DIR)
    else:
        print("Script finished with errors: Could not load data.")