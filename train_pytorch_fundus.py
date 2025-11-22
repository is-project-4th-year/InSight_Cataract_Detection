import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_DIR = "D:/BICS/insightweb/data/images/preprocessed_images/" 
LABEL_FILE = "D:/BICS/insightweb/data/images/full_df.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NEW_MODEL_NAME = "fundus_pytorch_model.pt" # Our new PyTorch model
# ---------------

# --- 1. Custom Focal Loss for PyTorch ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.float()
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# 2. Custom Dataset for Fundus Images 
class FundusDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filepath']
        label = row['Cataract']
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None 
        
def collate_fn(batch):
    """
    Custom collate_fn to filter out (None, None) pairs
    caused by file loading errors.
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: 
        # Return empty tensors if batch is empty after filtering
        return torch.Tensor(), torch.Tensor() 
    return torch.utils.data.dataloader.default_collate(batch)
# --- !!! END OF MOVE !!! ---

def load_data_df(label_file, data_dir):
    df = pd.read_csv(label_file)
    print("Parsing image-level labels...")
    
    left_df = df[['Left-Fundus', 'Left-Diagnostic Keywords']].copy()
    left_df.rename(columns={'Left-Fundus': 'filename'}, inplace=True)
    left_df['Cataract'] = left_df['Left-Diagnostic Keywords'].apply(lambda x: 1 if 'cataract' in str(x) else 0)
    
    right_df = df[['Right-Fundus', 'Right-Diagnostic Keywords']].copy()
    right_df.rename(columns={'Right-Fundus': 'filename'}, inplace=True)
    right_df['Cataract'] = right_df['Right-Diagnostic Keywords'].apply(lambda x: 1 if 'cataract' in str(x) else 0)
    
    all_images_df = pd.concat([left_df[['filename', 'Cataract']], right_df[['filename', 'Cataract']]]).dropna()
    all_images_df = all_images_df[all_images_df['filename'] != 'db_no_fovea.jpg'].reset_index(drop=True)
    all_images_df['filepath'] = all_images_df['filename'].apply(lambda x: os.path.join(data_dir, x))
    
    print(f"Verifying {len(all_images_df)} filepaths...")
    all_images_df['file_exists'] = all_images_df['filepath'].apply(os.path.exists)
    removed_count = len(all_images_df) - all_images_df['file_exists'].sum()
    if removed_count > 0:
        print(f"Warning: Removed {removed_count} missing file references.")
        all_images_df = all_images_df[all_images_df['file_exists'] == True].copy()
        
    all_images_df = all_images_df.drop(columns=['file_exists'])
    all_images_df['Cataract'] = all_images_df['Cataract'].astype(int)
    print("Label parsing and file verification complete.")
    return all_images_df

# --- 3. Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE[0], IMG_SIZE[1])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE[0], IMG_SIZE[1])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. Model (ResNet18) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1) 
model = model.to(device)

# --- 5. Data & Training ---
if __name__ == "__main__":
    all_df = load_data_df(LABEL_FILE, DATA_DIR)
    if all_df.empty: exit()

    print(all_df['Cataract'].value_counts())

    train_df, val_df = train_test_split(all_df, test_size=0.2, random_state=42, stratify=all_df['Cataract'])
    
    print(f"\nTraining images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    
    train_dataset = FundusDataset(train_df, transform=train_transform)
    val_dataset = FundusDataset(val_df, transform=val_transform)
    
    # num_workers=4 will now work correctly
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')

    print("\n--- STARTING PYTORCH TRAINING (ResNet18 + Focal Loss) ---")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if images.nelement() == 0: continue # Skip empty batches
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1) 
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images.nelement() == 0: continue # Skip empty batches
                images, labels = images.to(device), labels.to(device)
                labels = labels.view(-1, 1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                total += labels.size(0)
                val_corrects += (preds == labels).sum().item()
        
        if len(train_loader) > 0 and len(val_loader) > 0 and total > 0:
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = (val_corrects / total) * 100

            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), NEW_MODEL_NAME)
                print(f"   -> Model saved! New best val loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | No data loaded for training or validation.")
            break

    print("\n--- TRAINING COMPLETE ---")
    print(f"The best model has been saved as '{NEW_MODEL_NAME}'")