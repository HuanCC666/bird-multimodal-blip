# blip/dataset.py
import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class CUBBirdDataset(Dataset):
    def __init__(self, metadata_csv, description_json, image_root, processor, split='train'):
        self.df = pd.read_csv(metadata_csv)
        self.descriptions = json.load(open(description_json, 'r'))
        self.image_root = image_root
        self.processor = processor
        self.split = split

        # 筛选训练或测试集
        if split == 'train':
            self.df = self.df[self.df['is_train'] == 1]
        else:
            self.df = self.df[self.df['is_train'] == 0]

        self.samples = self.df.to_dict(orient='records')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_root, sample['img_path'])

        image = Image.open(img_path).convert("RGB")
        descriptions = self.descriptions[sample['img_path']]
        text = descriptions[idx % len(descriptions)]

        inputs = self.processor(images=image, text=text, return_tensors="pt", padding='max_length', truncation=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # ✅ 加上 label 字段
        inputs['label'] = torch.tensor(sample['class_id'], dtype=torch.long)

        return inputs

