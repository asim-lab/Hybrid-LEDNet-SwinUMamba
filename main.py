!pip install torch==2.0.1 torchvision==0.15.2
!pip install causal-conv1d==1.1.1
!pip install mamba-ssm
!pip install torchinfo timm numba
!pip install tensorflow numpy opencv-python-headless scikit-learn
!pip install einops
!pip install torchsummary torchprofile
!pip install dynamic_network_architectures
!pip install nnunetv2 monai
!pip install seg_metrics
!pip install timm mamba-ssm
!apt-get update
!apt-get install cuda-libraries-12-2
!pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu122
!pip install timm mamba-ssm torchinfo torchsummary torchprofile
!pip install dynamic_network_architectures monai nnunetv2
# Install PyTorch and torchvision
!pip install torch torchvision
# Install acvl_utils - specify the version that’s most compatible with your needs
!pip install acvl_utils==0.2
# Install blosc2 without versioning to avoid possible conflicts
!pip install blosc2
!pip install acvl_utils==0.2 blosc2==2.0.0 numpy==1.21.6
!pip install thop
!apt install python3-numpy

import sys
sys.path.append('/content/drive/MyDrive/Dataset/Swin-UMamba-main')

import os

# List the contents of the Swin-UMamba-main directory
swin_umamba_path = '/content/drive/MyDrive/Dataset/Swin-UMamba-main'
print("Contents of Swin-UMamba-main directory:")
print(os.listdir(swin_umamba_path))

# List the contents of the swin_umamba directory
swin_umamba_folder_path = '/content/drive/MyDrive/Dataset/Swin-UMamba-main/swin_umamba'
print("Contents of swin_umamba directory:")
print(os.listdir(swin_umamba_folder_path))

!pip install /content/drive/MyDrive/Dataset/Swin-UMamba-main/swin_umamba

!pip install -e /content/drive/MyDrive/Dataset/Swin-UMamba-main/swin_umamba

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/JiarunLiu/Swin-UMamba
# %cd Swin-UMamba/swin_umamba
!pip install -e .

import sys
import os

# Append the Swin-UMamba path to the system path
sys.path.append('/content/drive/MyDrive/Dataset/Swin-UMamba-main/swin_umamba/nnunetv2/nets')

# List the contents of the directory
swin_umamba_folder_path = '/content/drive/MyDrive/Dataset/Swin-UMamba-main/swin_umamba/nnunetv2/nets'
print("Contents of swin_umamba/nnunetv2/nets directory:")
print(os.listdir(swin_umamba_folder_path))

!nvcc --version

!nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

import sys
import numpy as np
import blosc2

# Monkey patch blosc2 to have an ndarray attribute as np.ndarray
blosc2.ndarray = np.ndarray

# Append model path and try to import
sys.path.append('/content/drive/MyDrive/Dataset/Swin-UMamba-main/swin_umamba/nnunetv2/nets')

from SwinUMamba import SwinUMamba

# Define model input channels and output channels
in_chans = 3  # RGB channels
out_chans = 1  # Binary segmentation

# Instantiate the model
model = SwinUMamba(in_chans=in_chans, out_chans=out_chans)
print(model)

import os
from PIL import Image

# Define paths
image_dir = '/content/drive/MyDrive/Dataset/PH2/trainx'
mask_dir = '/content/drive/MyDrive/Dataset/PH2/trainy'
edge_dir = '/content/drive/MyDrive/Dataset/PH2/trainy_edges'

# Define the target size and output directories
target_size = (256, 256)
resized_image_dir = '/content/drive/MyDrive/Dataset/PH2/resized_images'
resized_mask_dir = '/content/drive/MyDrive/Dataset/PH2/resized_masks'
resized_edge_dir = '/content/drive/MyDrive/Dataset/PH2/resized_edges'

# Function to resize images in a folder
def resize_images_in_folder(input_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            try:
                img = Image.open(file_path)
                img_resized = img.resize(target_size, Image.LANCZOS)
                img_resized.save(os.path.join(output_dir, filename))
                img.close()
            except Exception as e:
                print(f"Error resizing {file_path}: {e}")

# Resizing images, masks, and edges
resize_images_in_folder(image_dir, resized_image_dir, target_size)
resize_images_in_folder(mask_dir, resized_mask_dir, target_size)
resize_images_in_folder(edge_dir, resized_edge_dir, target_size)

print("Resizing complete. Resized images, masks, and edges saved to separate folders.")

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Function to load images, masks, and edges
def load_ph2_dataset(images_path, masks_path, edges_path, img_size=(256, 256)):
    images = []
    masks = []
    edges = []

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))
    edge_files = sorted(os.listdir(edges_path))

    # Print the number of files found in each folder
    print(f"Found {len(image_files)} images, {len(mask_files)} masks, and {len(edge_files)} edge files.")

    for idx, (img_file, mask_file, edge_file) in enumerate(zip(image_files, mask_files, edge_files)):
        # Load and resize images
        img = cv2.imread(os.path.join(images_path, img_file))
        img = cv2.resize(img, img_size)

        # Load and resize masks (grayscale)
        mask = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = np.expand_dims(mask, axis=-1)

        # Load and resize edges (grayscale)
        edge = cv2.imread(os.path.join(edges_path, edge_file), cv2.IMREAD_GRAYSCALE)
        edge = cv2.resize(edge, img_size)
        edge = np.expand_dims(edge, axis=-1)

        # Append to lists
        images.append(img)
        masks.append(mask)
        edges.append(edge)

        # Debug: print progress after every 20 images
        if idx % 20 == 0:
            print(f"Loaded {idx + 1} images, masks, and edges.")

    # Convert lists to numpy arrays and normalize images
    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32) / 255.0
    edges = np.array(edges, dtype=np.float32) / 255.0

    # Print shapes of data to verify loading
    print(f"Images shape: {images.shape}, Masks shape: {masks.shape}, Edges shape: {edges.shape}")

    return images, masks, edges

# Paths to the images, masks, and edges
images_path = '/content/drive/MyDrive/Dataset/PH2/resized_images'
masks_path = '/content/drive/MyDrive/Dataset/PH2/resized_masks'
edges_path = '/content/drive/MyDrive/Dataset/PH2/resized_edges'

# Load the dataset
X, y, edges = load_ph2_dataset(images_path, masks_path, edges_path)

# Debug: Print shape of X, y, and edges before splitting
print(f"Data shapes before split: X: {X.shape}, y: {y.shape}, edges: {edges.shape}")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
edges_train, edges_val = train_test_split(edges, test_size=0.2, random_state=42)

# Print dataset sizes for verification
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
print(f"Training edge maps: {len(edges_train)}, Validation edge maps: {len(edges_val)}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose
from sklearn.metrics import roc_curve, auc
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add the correct path for Swin U-Net Mamba model
import sys
sys.path.append('/content/drive/MyDrive/Swin-UMamba-main/swin_umamba/nnunetv2/nets')
from SwinUMamba import SwinUMamba

# Model input channels and number of output classes
in_chans = 4  # RGB + edge map
out_chans = 1  # Binary segmentation

# Instantiate the model
model = SwinUMamba(in_chans=in_chans, out_chans=out_chans)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, edge_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.edge_dir = edge_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.bmp', '_lesion.bmp'))
        edge_path = os.path.join(self.edge_dir, img_name.replace('.bmp', '_lesion_edges.bmp'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        edge = Image.open(edge_path).convert("L") if os.path.exists(edge_path) else Image.new("L", image.size)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            edge = self.transform(edge)

        image_with_edge = torch.cat([image, edge], dim=0)
        return image_with_edge, mask

# Paths
image_dir = '/content/drive/MyDrive/Dataset/PH2/trainx'
mask_dir = '/content/drive/MyDrive/Dataset/PH2/trainy'
edge_dir = '/content/drive/MyDrive/Dataset/PH2/trainy_edges'

# Transform
transform = Compose([Resize((256, 256)), ToTensor()])

# Dataset and DataLoader
dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, edge_dir=edge_dir, transform=transform)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training loop
for epoch in range(50):
    model.train()
    train_loss, train_acc = 0, 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.sigmoid(outputs)
        train_acc += (preds.round() == masks).float().mean().item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation loop
    model.eval()
    val_loss, val_acc, val_dice, val_se, val_sp = 0, 0, 0, 0, 0
    all_preds, all_masks = [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            val_acc += (preds.round() == masks).float().mean().item()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Plotting training and validation losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Prepare for ROC curve
all_masks_binary = (np.concatenate(all_masks) > 0.5).astype(int)
all_preds_prob = np.concatenate(all_preds)

# ROC curve
fpr, tpr, _ = roc_curve(all_masks_binary.flatten(), all_preds_prob.flatten())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Function to load images, masks, and edges
def load_isic_dataset(images_path, masks_path, edges_path, img_size=(256, 256)):
    images = []
    masks = []
    edges = []

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))
    edge_files = sorted(os.listdir(edges_path))

    # Print the number of files found in each folder
    print(f"Found {len(image_files)} images, {len(mask_files)} masks, and {len(edge_files)} edge files.")

    for idx, (img_file, mask_file, edge_file) in enumerate(zip(image_files, mask_files, edge_files)):
        # Load and resize images
        img = cv2.imread(os.path.join(images_path, img_file))
        img = cv2.resize(img, img_size)

        # Load and resize masks (grayscale)
        mask = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = np.expand_dims(mask, axis=-1)

        # Load and resize edges (grayscale)
        edge = cv2.imread(os.path.join(edges_path, edge_file), cv2.IMREAD_GRAYSCALE)
        edge = cv2.resize(edge, img_size)
        edge = np.expand_dims(edge, axis=-1)

        # Append to lists
        images.append(img)
        masks.append(mask)
        edges.append(edge)

        # Debug: print progress after every 20 images
        if idx % 20 == 0:
            print(f"Loaded {idx + 1} images, masks, and edges.")

    # Convert lists to numpy arrays and normalize images
    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32) / 255.0
    edges = np.array(edges, dtype=np.float32) / 255.0

    # Print shapes of data to verify loading
    print(f"Images shape: {images.shape}, Masks shape: {masks.shape}, Edges shape: {edges.shape}")

    return images, masks, edges

# Paths to the resized images, masks, and edges
images_path = '/content/drive/MyDrive/Dataset/ISIC_2017_resized/trainx_resized_256'
masks_path = '/content/drive/MyDrive/Dataset/ISIC_2017_resized/trainy_resized_256'
edges_path = '/content/drive/MyDrive/Dataset/ISIC_2017_resized/trainy_edges_resized_256'

# Load the dataset
X, y, edges = load_isic_dataset(images_path, masks_path, edges_path)

# Split the dataset into train (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp, edges_train, edges_temp = train_test_split(X, y, edges, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test, edges_val, edges_test = train_test_split(X_temp, y_temp, edges_temp, test_size=0.5, random_state=42)

# Print dataset sizes for verification
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Testing samples: {len(X_test)}")
print(f"Training edge maps: {len(edges_train)}, Validation edge maps: {len(edges_val)}, Testing edge maps: {len(edges_test)}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Add the correct path for Swin U-Net Mamba model
import sys
sys.path.append('/content/drive/MyDrive/Swin-UMamba-main/swin_umamba/nnunetv2/nets')
from SwinUMamba import SwinUMamba

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset Loader
class ISIC2017Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, edge_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.edge_dir = edge_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        edge_path = os.path.join(self.edge_dir, mask_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        edge = Image.open(edge_path).convert('L')

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            edge = self.mask_transform(edge)

        return image, mask, edge

# Paths
image_dir = '/content/drive/MyDrive/Dataset/ISIC_2017_resized/trainx_resized_256'
mask_dir = '/content/drive/MyDrive/Dataset/ISIC_2017_resized/trainy_resized_256'
edge_dir = '/content/drive/MyDrive/Dataset/ISIC_2017_resized/trainy_edges_resized_256'

# Stronger Data Augmentation
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.7),  # Flip with higher probability
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=40),  # Increase rotation
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.7, 1.3), shear=15),  # More extreme affine
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Random perspective distortion
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Blur to simulate noise
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.7),  # Consistent flipping with images
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=40),
    transforms.ToTensor()
])

# Datasets and DataLoader
dataset = ISIC2017Dataset(image_dir, mask_dir, edge_dir, image_transform=image_transform, mask_transform=mask_transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize model
in_chans = 4
out_chans = 1
model = SwinUMamba(in_chans=in_chans, out_chans=out_chans).to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Added weight decay for regularization

# Mixed Precision Training
scaler = GradScaler()

# Metrics
def dice_coeff(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def accuracy(pred, target):
    pred = torch.sigmoid(pred) > 0.5
    return (pred == target).float().mean()

# Training Loop
def train_model(model, criterion, optimizer, num_epochs=50):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, masks, edges in train_loader:
            images, masks, edges = images.to(device), masks.to(device), edges.to(device)
            inputs = torch.cat((images, edges), dim=1)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_acc += accuracy(outputs, masks).item()

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc / len(train_loader))

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, masks, edges in val_loader:
                images, masks, edges = images.to(device), masks.to(device), edges.to(device)
                inputs = torch.cat((images, edges), dim=1)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_acc += accuracy(outputs, masks).item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc / len(val_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

    # Plot Metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.show()

# Train Model
train_model(model, criterion, optimizer, num_epochs=50)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomRotation
from sklearn.metrics import roc_curve, auc
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the correct path for Swin U-Net Mamba model
import sys
sys.path.append('/content/drive/MyDrive/Swin-UMamba-main/swin_umamba/nnunetv2/nets')
from SwinUMamba import SwinUMamba

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, edge_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.edge_dir = edge_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_segmentation.png'))
        edge_path = os.path.join(self.edge_dir, img_name.replace('.jpg', '_segmentation.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        edge = Image.open(edge_path).convert("L") if os.path.exists(edge_path) else Image.new("L", image.size)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            edge = self.transform(edge)

        image_with_edge = torch.cat([image, edge], dim=0)
        return image_with_edge, mask

# Dataset paths
image_dir = '/content/drive/MyDrive/Dataset/ISIC_2017_resized/trainx_resized_256'
mask_dir = '/content/drive/MyDrive/Dataset/ISIC_2017_resized/trainy_resized_256'
edge_dir = '/content/drive/MyDrive/Dataset/ISIC_2017_resized/trainy_edges_resized_256'

# Data augmentation
transform = Compose([
    Resize((256, 256)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    ToTensor()
])

# Dataset and DataLoader
dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, edge_dir=edge_dir, transform=transform)

# Split the dataset into train (70%), validation (15%), and test (15%) sets
train_size, val_size, test_size = 1400, 300, 300
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize model
model = SwinUMamba(in_chans=4, out_chans=1).to('cuda')

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(50):
    model.train()
    train_loss, train_acc = 0, 0

    for images, masks in train_loader:
        images, masks = images.to('cuda'), masks.to('cuda')
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.sigmoid(outputs)
        train_acc += (preds.round() == masks).float().mean().item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation loop
    model.eval()
    val_loss, val_acc = 0, 0
    all_preds, all_masks = [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to('cuda'), masks.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            val_acc += (preds.round() == masks).float().mean().item()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch + 1}/50], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Plotting training and validation losses and accuracies
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Prepare for ROC curve
all_masks_binary = (np.concatenate(all_masks) > 0.5).astype(int)
all_preds_prob = np.concatenate(all_preds)

# ROC curve
fpr, tpr, _ = roc_curve(all_masks_binary.flatten(), all_preds_prob.flatten())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
