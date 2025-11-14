import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
import glob
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import re

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class LIDCSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.image_size = image_size
        self.samples = [] 
        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.to_tensor = transforms.ToTensor()

        search_pattern = os.path.join(
            self.root_dir, "LIDC-IDRI-*", "nodule-*", "images", "slice-*.png"   
        )
        image_paths = glob.glob(search_pattern)
        
        for image_path in image_paths:
            slice_filename = os.path.basename(image_path)
            nodule_dir = os.path.dirname(os.path.dirname(image_path))
            mask_path = os.path.join(nodule_dir, "mask-0", slice_filename)
            
            if os.path.exists(mask_path):
                self.samples.append((image_path, mask_path))

        if len(self.samples) == 0:
            raise ValueError(f"M2 Dataset empty. Found 0 samples in {self.root_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        image = self.resize(image)
        mask = self.resize(mask)
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask[mask > 0] = 1 
        return image, mask.float()

class MendeleyPickleDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        self.transform = transform 
        
        print(f"Loading M3 data from {pkl_path}...")
        try:
            df = pd.read_pickle(pkl_path)
        except Exception as e:
            raise IOError(f"Failed to load pickle file {pkl_path} with pandas. Error: {e}")
        
        self.all_samples = df.to_dict('records')

        if not isinstance(self.all_samples, list):
            raise TypeError(f"Loaded data from {pkl_path} was not a list.")
            
        print(f"Successfully loaded and converted {len(self.all_samples)} samples.")
            
        self.binarized_labels = []
        for sample in self.all_samples:
            label_str = str(sample['label1'])
            numeric_part = re.sub(r'[^0-9]', '', label_str)
            
            if numeric_part:
                label_lungrads = int(numeric_part)
                label_binary = 1 if label_lungrads > 2 else 0
                self.binarized_labels.append(label_binary)
            else:
                pass

    def __len__(self): 
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample_data = self.all_samples[idx]
        label_binary = self.binarized_labels[idx]
        image_hu = sample_data['hu_array']

        image_hu = np.clip(image_hu, -1000, 400)
        image_norm = (image_hu + 1000) / 1400  
        image_uint8 = (image_norm * 255).astype(np.uint8)
        
        image_pil = Image.fromarray(image_uint8).convert("L")
        
        if self.transform:
            image = self.transform(image_pil)
            
        return image, torch.tensor(label_binary, dtype=torch.long)

def get_data_loaders(task_name, task_config):
    data_dir = task_config['DATA_DIR']
    img_size = task_config['IMAGE_SIZE']
    batch_size = task_config.get('BATCH_SIZE', 32)
    seed = task_config.get('SEED', 42)
    
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) 
    ])
    transform_val_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) 
    ])

    if task_name == 'M1_CT_Classification':
        train_data = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        val_data = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_val_test)
        test_data = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform_val_test)
        num_classes = len(train_data.classes)

    elif task_name == 'M3_Rads_Classification':
        train_pkl_path = os.path.join(data_dir, "lung_cancer_train.pkl")
        test_pkl_path = os.path.join(data_dir, "lung_cancer_test.pkl")

        train_val_dataset = MendeleyPickleDataset(pkl_path=train_pkl_path, transform=transform_train)
        train_val_dataset_val_transforms = MendeleyPickleDataset(pkl_path=train_pkl_path, transform=transform_val_test)
        
        all_indices = list(range(len(train_val_dataset)))
        all_labels = train_val_dataset.binarized_labels

        train_indices, val_indices = train_test_split(
            all_indices, train_size=0.8, random_state=seed, stratify=all_labels
        )
        
        train_data = Subset(train_val_dataset, train_indices)
        val_data = Subset(train_val_dataset_val_transforms, val_indices)
        test_data = MendeleyPickleDataset(pkl_path=test_pkl_path, transform=transform_val_test)
        
        print(f"Dataset Split (M3_Rads_Classification): Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        num_classes = 2

    elif task_name == 'M2_LIDC_Segmentation':
        full_dataset = LIDCSegmentationDataset(root_dir=data_dir, image_size=img_size)
        
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size 
        
        train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size],
                                                       generator=torch.Generator().manual_seed(seed))
        
        print(f"Dataset Split (M2 LIDC): Total={len(full_dataset)}, Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        num_classes = 1
        
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_classes