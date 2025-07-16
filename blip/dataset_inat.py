# inat_dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class INatBirdDataset(Dataset):
    def __init__(self, metadata_csv, description_json, image_root, processor, split='train'):
        self.df = pd.read_csv(metadata_csv)
        
        # 根据 split 筛选
        if 'is_train' in self.df.columns:
            if split == 'train':
                self.df = self.df[self.df['is_train'] == 1]
            elif split == 'val':
                self.df = self.df[self.df['is_train'] == 0]
        
        self.df = self.df.reset_index(drop=True)
        self.image_root = image_root
        self.processor = processor
        
        # 加载描述 json
        with open(description_json, 'r') as f:
            self.descriptions = json.load(f)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row['file_name'])
        image = Image.open(image_path).convert("RGB")

        caption = self.descriptions.get(row['file_name'], "A bird.")
    
        inputs = self.processor(
              images=image,
              text=caption,
              padding='max_length',
              return_tensors="pt",
              max_length=128,
              truncation=True
            )

        return {
            "pixel_values": inputs["pixel_values"][0],   # 等价于 .squeeze(0)
            "input_ids":    inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
        }
