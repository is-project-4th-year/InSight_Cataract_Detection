import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# =============================
# Step 1: Load labels CSV
# =============================
df = pd.read_csv("D:/BICS/INSIGHT/data/slit_lamp/labels_slitlamp.csv")

# Split train/val/test
train_val, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['label'], random_state=42)

print("Train:", len(train), "Val:", len(val), "Test:", len(test))

# =============================
# Step 2: Define dataset class
# =============================
class SlitLampDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'filepath']
        label = self.data.loc[idx, 'label']
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# =============================
# Step 3: Define transforms & dataloaders
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = SlitLampDataset(train, transform=transform)
val_dataset   = SlitLampDataset(val, transform=transform)
test_dataset  = SlitLampDataset(test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("✅ DataLoaders ready")
